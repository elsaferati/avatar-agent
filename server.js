require('dotenv').config({ override: true });
const express = require('express');
const axios = require('axios');
const cors = require('cors');
const OpenAI = require('openai');
const { Pinecone } = require('@pinecone-database/pinecone');

const app = express();
app.use(cors());
app.use(express.json({ limit: '50mb' })); // Increased limit for Image Base64

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

async function getCompanyKnowledge(query) {
    if (!pc || !PINECONE_INDEX_NAME) return "";
    try {
        const embedding = await openai.embeddings.create({ model: "text-embedding-3-small", input: query });
        const index = pc.index(PINECONE_INDEX_NAME);
        const queryResponse = await index.query({ vector: embedding.data[0].embedding, topK: 2, includeMetadata: true });
        return queryResponse.matches.map(m => m.metadata.text || "").join("\n\n");
    } catch (e) { return ""; }
}

app.get('/simli-config', (req, res) => {
    conversationHistory = []; 
    res.json({ apiKey: SIMLI_API_KEY, faceID: SIMLI_FACE_ID });
});

app.post('/agent/speak', async (req, res) => {
    // We accept 'image' (base64) for vision, and 'type' to determine logic
    const { text, type, context, slideIndex, totalSlides, image } = req.body; 
    
    try {
        let messages = [];
        let finalScript = "";

        // --- SCENARIO 0: SHOULD WE ASK A QUESTION? (No TTS, JSON only) ---
        if (type === 'SHOULD_ASK') {
            const decisionCompletion = await openai.chat.completions.create({
                model: "gpt-4o",
                messages: [
                    {
                        role: "system",
                        content: `You decide if the presenter should pause to ask the audience for questions.
                        Return ONLY JSON: { "ask": true|false }
                        
                        Inputs you get:
                        - slide_index: ${slideIndex}
                        - total_slides: ${totalSlides}
                        - slide_text: "${text || ""}"
                        - slides_since_last_ask: ${req.body.slidesSinceLastAsk ?? "null"}
                        - recent_user_question: ${req.body.recentUserQuestion ?? false}

                        Rules:
                        - Never ask on title slide (slide 1) unless explicitly a Q&A slide.
                        - Avoid asking if the user just asked a question (recent_user_question=true).
                        - Prefer asking after dense/long slides or after every ~3-4 content slides.
                        - Avoid asking on the final slide unless it's a dedicated Q&A.
                        - Keep asks sparse: generally not more than once every 2 slides.
                        `
                    },
                    { role: "user", content: "Decide if we should ask for questions now." }
                ],
                response_format: { type: "json_object" },
                temperature: 0.2
            });

            const decision = JSON.parse(decisionCompletion.choices[0].message.content || "{}");
            return res.json({ ask: !!decision.ask });
        }

        // --- SCENARIO 1: THE BRAIN (Check User Intent) ---
        // This runs when the user speaks into the microphone. 
        // We check: Is this a question? Or do they want to move on?
        if (type === 'DECIDE_NEXT_MOVE') {
            const decisionCompletion = await openai.chat.completions.create({
                model: "gpt-4o",
                messages: [
                    {
                        role: "system",
                        content: `You are the presentation logic brain. 
                        Analyze the user's spoken input.
                        Return JSON ONLY: { "action": "ANSWER" | "RESUME" }

                        Rules:
                        - If the user asks a question, requests clarification, or makes a comment needing a reply -> action: "ANSWER"
                        - If the user says "No", "No questions", "Go ahead", "Continue", "Next", "That's clear" -> action: "RESUME"
                        `
                    },
                    { role: "user", content: text }
                ],
                response_format: { type: "json_object" },
                temperature: 0.0
            });

            const decision = JSON.parse(decisionCompletion.choices[0].message.content);
            // Return immediately. We don't generate audio for this step.
            return res.json(decision);
        }

        // --- SCENARIO 2: PRESENTING (With Vision) ---
        else if (type === 'PRESENT') {
            let styleInstruction = "";
            
            if (slideIndex === 1) {
                styleInstruction = `Start with: "Hi, I'm Elsa from PrimEx." Give a warm welcome, introduce the title visible on the slide, and add a 1-sentence teaser.`;
            } else if (slideIndex === totalSlides) {
                styleInstruction = `Summarize the key takeaway. Thank the audience. Ask if there are questions.`;
            } else {
                styleInstruction = `
                - Explain the visual content (charts, bullets, images).
                - Use transition phrases like "If you look at this graph..." or "As shown here...".
                - Keep it engaging.
                `;
            }

            messages = [
                {
                    role: "system",
                    content: `You are **Elsa**, the Lead Presenter for **PrimEx** and part of the PrimEx team. 
                    You can SEE the slide the user is looking at. 
                    ${styleInstruction}
                    RULES: Be punchy (Max 3 sentences). Speak as "we". 
                    At the end of your explanation, occasionally ask a quick check-in like "Does that make sense?"`
                },
                { 
                    role: "user", 
                    content: [
                        { type: "text", text: `I am showing Slide ${slideIndex}. The extracted text is: "${text}". Present this slide based on the image provided.` },
                        { type: "image_url", image_url: { url: image } } 
                    ]
                }
            ];

            const completion = await openai.chat.completions.create({
                model: "gpt-4o",
                messages: messages,
                max_tokens: 250,
                temperature: 0.7 
            });
            finalScript = completion.choices[0].message.content;
        } 
        
        // --- SCENARIO 3: ANSWERING (Conversational Loop) ---
        else if (type === 'ANSWER') {
            const pineconeData = await getCompanyKnowledge(text);
            messages = [
                {
                    role: "system",
                    content: `You are Elsa. You were interrupted by a question.
                    internal_knowledge: "${pineconeData}"
                    slide_context: "${context}"
                    
                    TASK:
                    1. Answer the user's question clearly and briefly.
                    2. IMMEDIATELY after answering, ask a short check-in question to see if they are done.
                    Example: "The cost is $5. Does that help?" or "We ship globally. Any other questions on that?"
                    `
                },
                ...conversationHistory, 
                { role: "user", content: text }
            ];

            const completion = await openai.chat.completions.create({
                model: "gpt-4o",
                messages: messages,
                max_tokens: 250,
                temperature: 0.7 
            });
            finalScript = completion.choices[0].message.content;
        }

        // --- SCENARIO 4: TTS ONLY (Direct Speech) ---
        // Used when we just want the Avatar to say a specific phrase without thinking (e.g., "Any questions?")
        else if (type === 'TTS_ONLY') {
            finalScript = text;
        }

        // --- COMMON: SAVE MEMORY & GENERATE AUDIO ---
        
        // Save to memory (unless it's just a TTS command)
        if (type !== 'TTS_ONLY') {
            conversationHistory.push({ role: "assistant", content: finalScript });
            if (conversationHistory.length > 6) conversationHistory = conversationHistory.slice(-6);
        }

        // Audio Generation (ElevenLabs)
        const ttsResp = await axios.post(
            `https://api.elevenlabs.io/v1/text-to-speech/${ELEVENLABS_VOICE_ID}?output_format=pcm_16000`,
            {
                text: finalScript,
                model_id: "eleven_turbo_v2",
                voice_settings: { stability: 0.4, similarity_boost: 0.8 } 
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
app.listen(PORT, () => console.log(`✅ Elsa V2 (Vision) Ready on http://localhost:${PORT}`));
