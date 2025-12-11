require('dotenv').config({ override: true });
const express = require('express');
const axios = require('axios');
const cors = require('cors');
const OpenAI = require('openai');
const { Pinecone } = require('@pinecone-database/pinecone');

const app = express();
app.use(cors());
app.use(express.json());

// --- CONFIGURATION ---
const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
const ELEVENLABS_API_KEY = process.env.ELEVENLABS_API_KEY;
const ELEVENLABS_VOICE_ID = process.env.ELEVENLABS_VOICE_ID || "21m00Tcm4TlvDq8ikWAM"; 
const SIMLI_API_KEY = process.env.SIMLI_API_KEY;
const SIMLI_FACE_ID = process.env.SIMLI_FACE_ID;
const PINECONE_API_KEY = process.env.PINECONE_API_KEY;
const PINECONE_INDEX_NAME = process.env.PINECONE_INDEX;

const openai = new OpenAI({ apiKey: OPENAI_API_KEY });

// Initialize Pinecone
let pc = null;
if (PINECONE_API_KEY) {
    pc = new Pinecone({ apiKey: PINECONE_API_KEY });
}

// Memory Storage
let conversationHistory = [];

// --- HELPER: GET KNOWLEDGE ---
async function getCompanyKnowledge(query) {
    if (!pc || !PINECONE_INDEX_NAME) return "";
    try {
        const embedding = await openai.embeddings.create({ model: "text-embedding-3-small", input: query });
        const index = pc.index(PINECONE_INDEX_NAME);
        const queryResponse = await index.query({ vector: embedding.data[0].embedding, topK: 2, includeMetadata: true });
        return queryResponse.matches.map(m => m.metadata.text || "").join("\n\n");
    } catch (e) { return ""; }
}

// 1. CONFIG ENDPOINT
app.get('/simli-config', (req, res) => {
    conversationHistory = []; // Reset memory
    res.json({ apiKey: SIMLI_API_KEY, faceID: SIMLI_FACE_ID });
});

// 2. THE PRESENTATION BRAIN
app.post('/agent/speak', async (req, res) => {
    // We now accept slideIndex and totalSlides to know "where" we are
    const { text, type, context, slideIndex, totalSlides } = req.body; 
    
    try {
        let messages = [];

        // --- SCENARIO A: PRESENTING (The "Human" Logic) ---
        if (type === 'PRESENT') {
            let styleInstruction = "";
            
            // 1. DYNAMIC BEHAVIOR BASED ON SLIDE NUMBER
            if (slideIndex === 1) {
                // FIRST SLIDE: Welcoming and setting the stage
                styleInstruction = `
                This is the FIRST slide. 
                - Start with a warm welcome ("Hello everyone, I'm Elsa from PrimEx, thanks for joining...").
                - Introduce the title of the presentation based on the text.
                - Give a brief 1-sentence teaser of what we will cover.
                - DO NOT just read the text. Act as the host.
                `;
            } else if (slideIndex === totalSlides) {
                // LAST SLIDE: Conclusion
                styleInstruction = `
                This is the LAST slide.
                - Summarize the key takeaway.
                - Thank the audience.
                - Ask if there are any questions.
                `;
            } else {
                // MIDDLE SLIDES: Storytelling
                styleInstruction = `
                This is a middle slide (Slide ${slideIndex} of ${totalSlides}).
                - Use transition phrases like "Moving on to...", "If you look here...", "What is interesting is...".
                - Act as if you are pointing at the slide. 
                - Explain the *significance* of the text, don't just repeat it.
                - Keep it engaging and high-energy.
                `;
            }

            messages = [
                {
                    role: "system",
                    content: `You are **Elsa**, the charismatic Lead Presenter for **PrimEx**.
                    
                    YOUR GOAL: You are NOT reading a script. You are presenting a deck to a live audience.
                    
                    ${styleInstruction}
                    
                    RULES:
                    - Be punchy. No long monologues. (Max 3 sentences).
                    - Speak as "we" (the company).
                    - Never say "Slide Number X". Just present the content.`
                },
                { role: "user", content: `Slide Content: "${text}"` }
            ];
        } 
        
        // --- SCENARIO B: ANSWERING QUESTIONS ---
        else if (type === 'ANSWER') {
            const pineconeData = await getCompanyKnowledge(text);
            messages = [
                {
                    role: "system",
                    content: `You are **Elsa**. You are currently presenting but just got interrupted by a question.
                    
                    CONTEXT:
                    - Internal Knowledge: "${pineconeData}"
                    - Slide Context: "${context}"
                    
                    Answer the question confidently but briefly. Then smoothly transition back to the presentation flow (e.g., "Great question. Now, back to the slide...").`
                },
                ...conversationHistory, 
                { role: "user", content: text }
            ];
        }

        // 1. Generate Script
        const completion = await openai.chat.completions.create({
            model: "gpt-4o-mini", // fast and capable enough for flow
            messages: messages,
            max_tokens: 200,
            temperature: 0.7 // Higher creativity for "Presenting" feel
        });
        
        const finalScript = completion.choices[0].message.content;

        // 2. Save Memory
        conversationHistory.push({ role: "assistant", content: finalScript });
        if (conversationHistory.length > 6) conversationHistory = conversationHistory.slice(-6);

        // 3. Audio Generation (ElevenLabs)
        const ttsResp = await axios.post(
            `https://api.elevenlabs.io/v1/text-to-speech/${ELEVENLABS_VOICE_ID}?output_format=pcm_16000`,
            {
                text: finalScript,
                model_id: "eleven_turbo_v2",
                voice_settings: { stability: 0.4, similarity_boost: 0.8 } // Lower stability = more emotion/range
            },
            {
                headers: { 'xi-api-key': ELEVENLABS_API_KEY, 'Content-Type': 'application/json' },
                responseType: 'arraybuffer'
            }
        );

        res.json({ 
            text: finalScript, 
            audio: Buffer.from(ttsResp.data).toString('base64') 
        });

    } catch (error) {
        console.error("Error:", error.message);
        res.status(500).json({ error: error.message });
    }
});

const PORT = 3000;
app.listen(PORT, () => console.log(`✅ Elsa is ready on http://localhost:${PORT}`));