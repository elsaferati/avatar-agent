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
    // NEW: We accept 'image' (base64)
    const { text, type, context, slideIndex, totalSlides, image } = req.body; 
    
    try {
        let messages = [];

        // --- SCENARIO A: PRESENTING (With Vision) ---
        if (type === 'PRESENT') {
            let styleInstruction = "";
            
            if (slideIndex === 1) {
                styleInstruction = `Start with a warm welcome. Introduce the title visible on the slide. Give a 1-sentence teaser.`;
            } else if (slideIndex === totalSlides) {
                styleInstruction = `Summarize the key takeaway. Thank the audience. Ask if there are questions.`;
            } else {
                styleInstruction = `
                - Explain the visual content of the slide (charts, bullets, images).
                - Use transition phrases like "If you look at this graph..." or "As shown here...".
                - Keep it engaging.
                
                TEACHER MODE: Occasionally (every 2-3 slides) ask a rhetorical question to check understanding, like "Does that make sense?" or "Pretty interesting, right?"
                `;
            }

            // VISION PAYLOAD
            messages = [
                {
                    role: "system",
                    content: `You are **Elsa**, the Lead Presenter for **PrimEx**. 
                    You can SEE the slide the user is looking at. 
                    ${styleInstruction}
                    RULES: Be punchy (Max 3 sentences). Speak as "we".`
                },
                { 
                    role: "user", 
                    content: [
                        { type: "text", text: `I am showing Slide ${slideIndex}. The extracted text is: "${text}". Present this slide based on the image provided.` },
                        { type: "image_url", image_url: { url: image } } // Pass the image to GPT-4o
                    ]
                }
            ];
        } 
        
        // --- SCENARIO B: ANSWERING ---
        else if (type === 'ANSWER') {
            const pineconeData = await getCompanyKnowledge(text);
            messages = [
                {
                    role: "system",
                    content: `You are Elsa. You were presenting but got interrupted.
                    internal_knowledge: "${pineconeData}"
                    slide_context: "${context}"
                    Answer briefly and transition back to the presentation.`
                },
                ...conversationHistory, 
                { role: "user", content: text }
            ];
        }

        const completion = await openai.chat.completions.create({
            model: "gpt-4o", // SWITCHED TO GPT-4o FOR VISION
            messages: messages,
            max_tokens: 250,
            temperature: 0.7 
        });
        
        const finalScript = completion.choices[0].message.content;

        // Save Memory
        conversationHistory.push({ role: "assistant", content: finalScript });
        if (conversationHistory.length > 6) conversationHistory = conversationHistory.slice(-6);

        // Audio Generation
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