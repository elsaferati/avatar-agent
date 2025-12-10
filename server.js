// Load .env and let it override any existing env vars so local edits take effect
require('dotenv').config({ override: true });
const express = require('express');
const axios = require('axios');
const cors = require('cors');
const OpenAI = require('openai');
// [ADDED] Import Pinecone
const { Pinecone } = require('@pinecone-database/pinecone');

const app = express();
app.use(cors());
app.use(express.json());

// --- SIMLI CONFIG ---
const SIMLI_API_KEY = process.env.SIMLI_API_KEY || "";
const SIMLI_BASE = "https://api.simli.ai";
const SIMLI_FACE_ID = process.env.SIMLI_FACE_ID || "";
const SIMLI_TTS_PROVIDER = process.env.SIMLI_TTS_PROVIDER || "ElevenLabs";
const SIMLI_VOICE_ID = process.env.SIMLI_VOICE_ID || "";
const SIMLI_TTS_API_KEY = process.env.SIMLI_TTS_API_KEY || null; 
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";

// [ADDED] Pinecone Config & Memory
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX;
let conversationHistory = []; // Stores chat history

const openai = OPENAI_API_KEY ? new OpenAI({ apiKey: OPENAI_API_KEY }) : null;

// Helper: resilient chat completion with model fallback
async function callOpenAIChat(messages, max_tokens = 150) {
    if (!openai) throw new Error('OpenAI not configured');
    const candidateModels = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"];
    let lastErr = null;
    for (const model of candidateModels) {
        try {
            const completion = await openai.chat.completions.create({
                model,
                messages,
                max_tokens
            });
            return { completion, model };
        } catch (e) {
            console.warn(`OpenAI model ${model} failed:`, e.message || e);
            lastErr = e;
        }
    }
    throw lastErr || new Error('All OpenAI model calls failed');
}

// [ADDED] Initialize Pinecone (Safe check)
let pc = null;
if (PINECONE_API_KEY) {
    pc = new Pinecone({ apiKey: PINECONE_API_KEY });
} else {
    console.warn("⚠️ PINECONE_API_KEY missing. Memory will work, but Company Knowledge will be empty.");
}

// [ADDED] Helper to Search Pinecone
async function getCompanyKnowledge(query) {
    if (!pc || !PINECONE_INDEX_NAME) return "";
    try {
        const embedding = await openai.embeddings.create({
            model: "text-embedding-3-small", 
            input: query,
        });
        const index = pc.index(PINECONE_INDEX_NAME);
        const queryResponse = await index.query({
            vector: embedding.data[0].embedding,
            topK: 3, 
            includeMetadata: true 
        });
        return queryResponse.matches.map(m => m.metadata.text || "").join("\n\n");
    } catch (e) {
        console.error("Pinecone Error:", e.message);
        return "";
    }
}

// Helper to determine TTS settings
const getTTSPayload = () => {
    const payload = {};
    if (SIMLI_TTS_API_KEY) {
        payload.ttsProvider = SIMLI_TTS_PROVIDER;
        payload.ttsAPIKey = SIMLI_TTS_API_KEY;
        if (SIMLI_TTS_PROVIDER.toLowerCase() === "elevenlabs" && SIMLI_VOICE_ID) {
            payload.voiceId = SIMLI_VOICE_ID;
        }
    }
    return payload;
};

// 1. CONNECT AVATAR (Your exact logic, just added memory reset)
app.post('/simli/start-session', async (req, res) => {
    try {
        // [ADDED] Reset memory on new session
        conversationHistory = [];

        const simliPayload = {
            apiKey: SIMLI_API_KEY,
            faceId: SIMLI_FACE_ID,
            firstMessage: "Hello! I am ready to present.", 
            ...getTTSPayload()
        };

        const resp = await axios.post(
            `${SIMLI_BASE}/auto/start/configurable`,
            simliPayload,
            { headers: { 'Content-Type': 'application/json' } }
        );
        
        console.log("Session started:", resp.data);
        res.json(resp.data); 
    } catch (error) {
        console.error("Start Session Error:", error.response?.data || error.message);
        res.status(500).json(error.response?.data || error.message);
    }
});

// =========================================================
// 2. PRESENTATION ENDPOINT (The "Florent" Persona)
// =========================================================
app.post('/agent/present', async (req, res) => {
    if (!openai) return res.status(500).json({ error: "OpenAI not configured" });
    
    const { slide_text } = req.body;
    
    try {
        // 1. Generate Script (Enforcing Persona Here)
        const messages = [
            { 
                role: "system", 
                content: `You are **Florent**, a senior team member and representative of **PrimEx**. 
                    You are currently presenting a slide deck to an audience.
                    
                    INSTRUCTIONS:
                    - Summarize the slide text provided by the user in 2-3 engaging sentences.
                    - Always speak as "we" (representing the company).
                    - Be professional, confident, and clear.
                    - Do not say "Next slide" yet, just cover the content.` 
            },
            { role: "user", content: slide_text }
        ];

        const { completion, model: usedModel } = await callOpenAIChat(messages, 150);
        const speech = completion.choices[0].message.content;
        console.log(`Used OpenAI model for /agent/present: ${usedModel}`);

        // [CRITICAL] Save as Florent in memory so he remembers he said this
        conversationHistory.push({ role: "assistant", content: `(Florent presented): ${speech}` });

        // 2. Send to Simli
        const simliPayload = {
            apiKey: SIMLI_API_KEY,
            faceId: SIMLI_FACE_ID,
            firstMessage: speech,
            ...getTTSPayload()
        };

        // Attempt to call Simli, but don't fail the whole request if Simli is unavailable.
        let simliRespData = null;
        try {
            const simliResp = await axios.post(
                `${SIMLI_BASE}/auto/start/configurable`,
                simliPayload,
                { headers: { 'Content-Type': 'application/json' } }
            );
            simliRespData = simliResp.data;
        } catch (e) {
            console.warn('Simli call failed in /agent/present:', e.response?.data || e.message || e);
        }

        res.json({ speech, simli: simliRespData });
    } catch (error) {
        console.error("Agent Present Error:", error.response?.data || error.message);
        res.status(500).json({ error: error.message });
    }
});

// =========================================================
// 3. INTERACTION ENDPOINT (Pinecone + Persona + Debugging)
// =========================================================
app.post('/agent/interact', async (req, res) => {
    if (!openai) return res.status(500).json({ error: "OpenAI not configured" });
    
    const { user_question, slide_context } = req.body;

    try {
        console.log(`\n💬 Received Question: "${user_question}"`);

        // 1. Get Knowledge from Pinecone
        const companyInfo = await getCompanyKnowledge(user_question);
        
        // DEBUG: Check if Pinecone found anything
        if(companyInfo) console.log("✅ Pinecone Context Found (Preview):", companyInfo.substring(0, 50) + "...");
        else console.log("⚠️ No Pinecone Context found.");

        // 2. Build Message Chain (Strict "Florent" Persona)
        const messages = [
            { 
                role: "system", 
                content: `You are **Florent**, a senior representative and core team member of **PrimEx**. 
                You are currently giving a presentation, but the user has interrupted you with a question.

                ### KNOWLEDGE SOURCES (In order of priority):
                1. **YOUR BRAIN (Internal Database):** 
                """${companyInfo}"""
                *(Use this first. If the answer is here, ignore the slide.)*

                2. **CURRENT SLIDE:** 
                """${slide_context}"""
                *(Use this only if the database doesn't have the answer.)*

                ### INSTRUCTIONS:
                - **Identity:** Your name is Florent. Always speak as "we" (PrimEx).
                - **Tone:** Professional, helpful, and knowledgeable.
                - **Flow:** Answer the question concisely. 
                - **Closing:** End your answer with a subtle transition back to the presentation (e.g., "I hope that clears that up. Moving on..." or "That is a key part of our strategy.").
                ` 
            },
            ...conversationHistory, // Inject Memory
            { 
                role: "user", 
                content: `Question: "${user_question}"` 
            }
        ];

        // DEBUG: See exactly what OpenAI gets
        console.log("📨 Sending Prompt to OpenAI...");

        // 3. Generate Answer (use resilient OpenAI call)
        const { completion, model: usedModel } = await callOpenAIChat(messages, 150);
        const answer = completion.choices[0].message.content;
        console.log(`Used OpenAI model for /agent/interact: ${usedModel}`);

        console.log(`🤖 Florent Answered: "${answer}"`);

        // 4. Save to Memory
        conversationHistory.push({ role: "user", content: user_question });
        conversationHistory.push({ role: "assistant", content: answer });
        
        // Keep memory manageable (last 10 turns)
        if (conversationHistory.length > 10) conversationHistory = conversationHistory.slice(-10);

        // 5. Send to Simli
        const simliPayload = {
            apiKey: SIMLI_API_KEY,
            faceId: SIMLI_FACE_ID,
            firstMessage: answer,
            ...getTTSPayload()
        };

        // Attempt to call Simli, but don't fail the whole request if Simli is unavailable.
        let simliRespData = null;
        try {
            const simliResp = await axios.post(
                `${SIMLI_BASE}/auto/start/configurable`,
                simliPayload,
                { headers: { 'Content-Type': 'application/json' } }
            );
            simliRespData = simliResp.data;
        } catch (e) {
            console.warn('Simli call failed in /agent/interact:', e.response?.data || e.message || e);
        }

        res.json({ answer, simli: simliRespData });
    } catch (error) {
        console.error("Agent Interact Error:", error.response?.data || error.message);
        res.status(500).json({ error: error.message });
    }
});

// --- DEBUG ENDPOINT ---
app.get('/test-pinecone', async (req, res) => {
    const query = req.query.q || "What does this company do?"; // Default question
    
    try {
        console.log(`🧪 Testing Pinecone with query: "${query}"`);
        
        // 1. Generate Embedding
        const modelName = "text-embedding-3-small"; // Make sure this matches your upload!
        const embedding = await openai.embeddings.create({
            model: modelName, 
            input: query,
        });

        // 2. Query Pinecone directly
        if (!pc || !PINECONE_INDEX_NAME) {
            return res.status(400).json({ error: 'Pinecone not configured (missing API key or index name)' });
        }
        const index = pc.index(PINECONE_INDEX_NAME);
        const queryResponse = await index.query({
            vector: embedding.data[0].embedding,
            topK: 3, 
            includeMetadata: true 
        });

        // 3. Return the raw data to the browser
        res.json({
            status: "success",
            matches_found: queryResponse.matches.length,
            first_match_score: queryResponse.matches[0]?.score,
            raw_matches: queryResponse.matches // This lets you see the metadata structure
        });

    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

const PORT = 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));