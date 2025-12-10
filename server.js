// Load .env and let it override any existing env vars so local edits take effect
require('dotenv').config({ override: true });
const express = require('express');
const axios = require('axios');
const cors = require('cors');
const OpenAI = require('openai');

const app = express();
app.use(cors());
app.use(express.json());

// --- SIMLI CONFIG ---
const SIMLI_API_KEY = process.env.SIMLI_API_KEY || "";
const SIMLI_BASE = "https://api.simli.ai";
const SIMLI_FACE_ID = process.env.SIMLI_FACE_ID || "";
const SIMLI_TTS_PROVIDER = process.env.SIMLI_TTS_PROVIDER || "ElevenLabs";
const SIMLI_VOICE_ID = process.env.SIMLI_VOICE_ID || "";
const SIMLI_TTS_API_KEY = process.env.SIMLI_TTS_API_KEY || null; // Only if using external TTS key
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";

const openai = OPENAI_API_KEY ? new OpenAI({ apiKey: OPENAI_API_KEY }) : null;

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

// 1. CONNECT AVATAR (Changed to use Simli Auto so we get a URL)
app.post('/simli/start-session', async (req, res) => {
    try {
        const simliPayload = {
            apiKey: SIMLI_API_KEY,
            faceId: SIMLI_FACE_ID,
            firstMessage: "Hello! I am ready to present.", // Initial greeting
            ...getTTSPayload()
        };

        const resp = await axios.post(
            `${SIMLI_BASE}/auto/start/configurable`,
            simliPayload,
            { headers: { 'Content-Type': 'application/json' } }
        );
        
        console.log("Session started:", resp.data);
        res.json(resp.data); // Returns { meetingUrl: "..." }
    } catch (error) {
        console.error("Start Session Error:", error.response?.data || error.message);
        res.status(500).json(error.response?.data || error.message);
    }
});

// 2. PRESENTATION ENDPOINT
app.post('/agent/present', async (req, res) => {
    if (!openai) return res.status(500).json({ error: "OpenAI not configured" });
    
    const { slide_text } = req.body;
    
    try {
        // 1. Generate Script
        const completion = await openai.chat.completions.create({
            model: "gpt-4o-mini",
            messages: [
                { role: "system", content: "You are a charismatic presenter. Summarize this slide in 2-3 engaging sentences." },
                { role: "user", content: slide_text }
            ],
            max_tokens: 150
        });
        const speech = completion.choices[0].message.content;

        // 2. Send to Simli
        const simliPayload = {
            apiKey: SIMLI_API_KEY,
            faceId: SIMLI_FACE_ID,
            firstMessage: speech,
            ...getTTSPayload()
        };

        const simliResp = await axios.post(
            `${SIMLI_BASE}/auto/start/configurable`,
            simliPayload,
            { headers: { 'Content-Type': 'application/json' } }
        );

        res.json({ speech, simli: simliResp.data });
    } catch (error) {
        console.error("Agent Present Error:", error.response?.data || error.message);
        res.status(500).json({ error: error.message });
    }
});

// 3. INTERACTION ENDPOINT
app.post('/agent/interact', async (req, res) => {
    if (!openai) return res.status(500).json({ error: "OpenAI not configured" });
    
    const { user_question, slide_context } = req.body;

    try {
        // 1. Generate Answer
        const completion = await openai.chat.completions.create({
            model: "gpt-4o-mini",
            messages: [
                { role: "system", content: "You are a helpful presenter answering a question about the current slide. Be brief." },
                { role: "user", content: `Context: ${slide_context}\n\nQuestion: ${user_question}` }
            ],
            max_tokens: 150
        });
        const answer = completion.choices[0].message.content;

        // 2. Send to Simli
        const simliPayload = {
            apiKey: SIMLI_API_KEY,
            faceId: SIMLI_FACE_ID,
            firstMessage: answer,
            ...getTTSPayload()
        };

        const simliResp = await axios.post(
            `${SIMLI_BASE}/auto/start/configurable`,
            simliPayload,
            { headers: { 'Content-Type': 'application/json' } }
        );

        res.json({ answer, simli: simliResp.data });
    } catch (error) {
        console.error("Agent Interact Error:", error.response?.data || error.message);
        res.status(500).json({ error: error.message });
    }
});

const PORT = 3000;
app.listen(PORT, () => console.log(`Server running on port ${PORT}`));