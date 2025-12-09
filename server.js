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
const SIMLI_AGENT_ID = process.env.SIMLI_AGENT_ID || "";
const SIMLI_FACE_ID = process.env.SIMLI_FACE_ID || "";
const SIMLI_TTS_PROVIDER = process.env.SIMLI_TTS_PROVIDER || "ElevenLabs";
const SIMLI_VOICE_ID = process.env.SIMLI_VOICE_ID || "";
const SIMLI_TTS_API_KEY = process.env.SIMLI_TTS_API_KEY || null;
const OPENAI_API_KEY = process.env.OPENAI_API_KEY || "";
const OPENAI_PROJECT = process.env.OPENAI_PROJECT || "";
const openai = OPENAI_API_KEY
    ? new OpenAI(OPENAI_PROJECT ? { apiKey: OPENAI_API_KEY, project: OPENAI_PROJECT } : { apiKey: OPENAI_API_KEY })
    : null;

// Only include TTS provider details when an external TTS key is provided.
const resolvedTTSProvider = SIMLI_TTS_PROVIDER;
const useElevenLabs = resolvedTTSProvider?.toLowerCase() === "elevenlabs";

function requireKey(res) {
    if (!SIMLI_API_KEY) {
        res.status(400).json({ error: "SIMLI_API_KEY is missing. Set it in .env" });
        return false;
    }
    return true;
}

function requireSimliConfig(res) {
    if (!SIMLI_API_KEY || !SIMLI_FACE_ID) {
        res.status(400).json({ error: "Simli config missing (SIMLI_API_KEY or SIMLI_FACE_ID)." });
        return false;
    }
    return true;
}

// Request logger (helps debug which routes are hit)
app.use((req, _res, next) => {
    console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
    next();
});

// Start a Simli LiveKit session (WebRTC)
// Docs: https://docs.simli.com/api-reference/endpoint/webrtc/startAudioToVideoSession
app.post('/simli/start-session', async (req, res) => {
    if (!requireKey(res)) return;
    try {
        const payload = req.body || {};
        // Ensure apiKey is present in body (Simli requires it)
        if (!payload.apiKey) payload.apiKey = SIMLI_API_KEY;
        const resp = await axios.post(
            `${SIMLI_BASE}/startAudioToVideoSession`,
            payload,
            { headers: { 'api-key': SIMLI_API_KEY, 'Content-Type': 'application/json' } }
        );
        res.json(resp.data);
    } catch (error) {
        console.error("Simli start-session error:", error.response ? error.response.data : error.message);
        res.status(500).json(error.response ? error.response.data : error.message);
    }
});

// Fetch ICE servers for WebRTC (Simli expects apiKey in body)
// Docs: https://docs.simli.com/api-reference/endpoint/webrtc/getIceServers
app.all('/simli/ice-servers', async (_req, res) => {
    if (!requireKey(res)) return;
    try {
        const resp = await axios.post(
            `${SIMLI_BASE}/getIceServers`,
            { apiKey: SIMLI_API_KEY },
            { headers: { 'api-key': SIMLI_API_KEY, 'Content-Type': 'application/json' } }
        );
        res.json(resp.data);
    } catch (error) {
        console.error("Simli get-ice-servers error:", error.response ? error.response.data : error.message);
        res.status(500).json(error.response ? error.response.data : error.message);
    }
});

// Proxy for Simli Auto configurable: sends text as firstMessage and returns the URL
app.post('/simli/talk', async (req, res) => {
    if (!requireKey(res) || !requireSimliConfig(res)) return;
    const { text } = req.body || {};
    if (!text) return res.status(400).json({ error: "text is required" });
    try {
        const talkPayload = {
            apiKey: SIMLI_API_KEY,
            faceId: SIMLI_FACE_ID,
            language: "en",
            firstMessage: text,
            createTranscript: false
        };
        if (SIMLI_TTS_API_KEY) {
            talkPayload.ttsProvider = resolvedTTSProvider;
            talkPayload.ttsAPIKey = SIMLI_TTS_API_KEY;
            if (useElevenLabs && SIMLI_VOICE_ID) {
                talkPayload.voiceId = SIMLI_VOICE_ID;
            }
        }
        const resp = await axios.post(
            `${SIMLI_BASE}/auto/start/configurable`,
            talkPayload,
            { headers: { 'Content-Type': 'application/json' } }
        );
        res.json(resp.data);
    } catch (error) {
        const payload = error.response ? error.response.data : error.message;
        console.error("Simli talk error:", JSON.stringify(payload, null, 2));
        res.status(error.response?.status || 500).json(payload);
    }
});

// Agent endpoint: summarize slide text with OpenAI, then send to Simli talk
app.post('/agent/present', async (req, res) => {
    if (!requireKey(res) || !requireSimliConfig(res)) return;
    if (!OPENAI_API_KEY || !openai) return res.status(400).json({ error: "OPENAI_API_KEY missing in .env" });
    const { slide_text = "" } = req.body || {};

    try {
        const prompt = `You are an engaging presenter. Summarize this slide in 3-5 spoken sentences. Be concise, friendly, and avoid reading every bullet. Slide text: ${slide_text}`;
        const completion = await openai.chat.completions.create({
            model: "gpt-4o-mini",
            messages: [
                { role: "system", content: "You are a helpful, concise presenter." },
                { role: "user", content: prompt }
            ],
            max_tokens: 180,
            temperature: 0.7
        });
        const speech = completion.choices?.[0]?.message?.content?.trim() || "Here's your slide.";

        // Send to Simli
        const agentTalkPayload = {
            apiKey: SIMLI_API_KEY,
            faceId: SIMLI_FACE_ID,
            language: "en",
            firstMessage: speech,
            createTranscript: false
        };
        if (SIMLI_TTS_API_KEY) {
            agentTalkPayload.ttsProvider = resolvedTTSProvider;
            agentTalkPayload.ttsAPIKey = SIMLI_TTS_API_KEY;
            if (useElevenLabs && SIMLI_VOICE_ID) {
                agentTalkPayload.voiceId = SIMLI_VOICE_ID;
            }
        }
        const talkResp = await axios.post(
            `${SIMLI_BASE}/auto/start/configurable`,
            agentTalkPayload,
            { headers: { 'Content-Type': 'application/json' } }
        );

        console.log('Agent present success:', { speech: speech.slice(0, 120), simli: talkResp.data });
        res.json({ speech, simli: talkResp.data });
    } catch (error) {
        const payload = error.response ? { status: error.response.status, data: error.response.data } : error.message;
        console.error("Agent present error:", JSON.stringify(payload, null, 2));
        res.status(error.response?.status || 500).json(error.response ? error.response.data : error.message);
    }
});

// Simple healthcheck
app.get('/health', (_req, res) => res.json({ ok: true }));

const PORT = process.env.PORT || 3000;
const server = app.listen(PORT, () => console.log(`Server running on port ${PORT}`));
server.on('close', () => console.log('HTTP server closed'));
server.on('error', (err) => {
    console.error('HTTP server error:', err);
});

// Debug: show active handles after startup
setTimeout(() => {
    const handles = process._getActiveHandles().map(h => h.constructor.name);
    console.log('Active handles after start:', handles);
}, 500);

process.on('exit', (code) => {
    console.log(`Process exiting with code ${code}`);
});
process.on('SIGINT', () => {
    console.log('Received SIGINT, shutting down...');
    server.close(() => process.exit(0));
});
process.on('uncaughtException', (err) => {
    console.error('Uncaught exception:', err);
    process.exit(1);
});
process.on('unhandledRejection', (reason) => {
    console.error('Unhandled rejection:', reason);
    process.exit(1);
});
