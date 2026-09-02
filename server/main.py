"""
Dog Talk - Main Server
FastAPI + WebSocket server that receives frames/audio from the Android app
and returns real-time behavior analysis.
"""
import asyncio
import base64
import json
import logging
import time
import sys
import os
import numpy as np
import cv2
from io import BytesIO

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from config import config
from vision.dog_detector import DogDetector
from vision.body_language import BodyLanguageAnalyzer
from audio.sound_classifier import SoundClassifier
from brain.knowledge_base import DogKnowledgeBase
from brain.behavior_engine import BehaviorEngine
from brain.llm_interpreter import LLMInterpreter

# Reconfigure stdout/stderr to UTF-8 to prevent encoding errors on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("dogtalk")

# ─── FastAPI App ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Dog Talk AI Server",
    description="Real-time dog behavior analysis server",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ───────────────────────────────────────────────────────────────

class ServerState:
    """Holds initialized ML models and pipeline components."""
    def __init__(self):
        self.detector: DogDetector = None
        self.body_analyzer: BodyLanguageAnalyzer = None
        self.sound_classifier: SoundClassifier = None
        self.knowledge_base: DogKnowledgeBase = None
        self.behavior_engine: BehaviorEngine = None
        self.llm: LLMInterpreter = None
        self.ready = False
        self.connected_clients = 0
        self.total_frames_processed = 0
        self.avg_processing_time_ms = 0.0

state = ServerState()

# ─── Startup / Shutdown ────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Initialize all ML models and pipelines."""
    logger.info("=" * 60)
    logger.info("🐕 Dog Talk AI Server Starting...")
    logger.info("=" * 60)
    
    try:
        # 1. Dog Detector (Vision)
        logger.info("Loading dog detector (YOLOv8)...")
        state.detector = DogDetector(
            model_path=config.yolo_model_path if os.path.exists(config.yolo_model_path) else None,
            confidence=config.detection_confidence,
            pose_confidence=config.pose_confidence
        )
        logger.info("✅ Dog detector ready")

        # 2. Body Language Analyzer
        state.body_analyzer = BodyLanguageAnalyzer()
        logger.info("✅ Body language analyzer ready")

        # 3. Sound Classifier (Audio)
        logger.info("Loading sound classifier (YAMNet)...")
        try:
            state.sound_classifier = SoundClassifier(
                custom_model_path=config.sound_classifier_path if os.path.exists(config.sound_classifier_path) else None,
                sample_rate=config.audio_sample_rate
            )
            logger.info("✅ Sound classifier ready")
        except Exception as e:
            logger.warning(f"⚠️ Sound classifier unavailable: {e}")
            state.sound_classifier = None

        # 4. Knowledge Base
        logger.info("Loading dog behavior knowledge base...")
        state.knowledge_base = DogKnowledgeBase(config.knowledge_base_path)
        logger.info("✅ Knowledge base ready")

        # 5. LLM Interpreter (optional)
        if config.llm_enabled:
            logger.info("Checking LLM availability (Ollama)...")
            state.llm = LLMInterpreter(
                host=config.ollama_host,
                model=config.ollama_model,
                timeout=config.llm_timeout
            )
            if state.llm.available:
                logger.info(f"✅ LLM ready ({state.llm.model})")
            else:
                logger.info("ℹ️ LLM not available — using knowledge base only")
        
        # 6. Behavior Engine (combines everything)
        state.behavior_engine = BehaviorEngine(
            knowledge_base=state.knowledge_base,
            llm_interpreter=state.llm
        )
        logger.info("✅ Behavior engine ready")

        state.ready = True
        logger.info("=" * 60)
        logger.info(f"🐕 Dog Talk Server ready on {config.host}:{config.port}")
        logger.info(f"   WebSocket: ws://{config.host}:{config.port}/ws")
        logger.info("=" * 60)

    except Exception as e:
        logger.error(f"❌ Server startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown():
    logger.info("🐕 Dog Talk Server shutting down...")


# ─── REST Endpoints ─────────────────────────────────────────────────────────────

@app.get("/")
async def root():
    return {"name": "Dog Talk AI Server", "version": "1.0.0", "ready": state.ready}


@app.get("/health")
async def health():
    return JSONResponse({
        "status": "ok" if state.ready else "initializing",
        "detector": state.detector is not None,
        "audio": state.sound_classifier is not None,
        "llm": state.llm.available if state.llm else False,
        "knowledge_base": state.knowledge_base is not None,
        "connected_clients": state.connected_clients,
        "total_frames": state.total_frames_processed,
        "avg_processing_ms": round(state.avg_processing_time_ms, 1)
    })


@app.get("/discover")
async def discover():
    """Service discovery endpoint for the Android app."""
    return JSONResponse({
        "service": "dogtalk",
        "version": "1.0.0",
        "ws_endpoint": f"ws://{config.host}:{config.port}/ws",
        "capabilities": {
            "vision": state.detector is not None,
            "audio": state.sound_classifier is not None,
            "llm": state.llm.available if state.llm else False
        }
    })


# ─── WebSocket Handler ──────────────────────────────────────────────────────────

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Main WebSocket handler for real-time analysis."""
    await websocket.accept()
    state.connected_clients += 1
    client_id = id(websocket)
    logger.info(f"📱 Client connected (total: {state.connected_clients})")
    
    # Per-client body language analyzer (has temporal state)
    body_analyzer = BodyLanguageAnalyzer()
    frame_count = 0

    try:
        while True:
            # Receive message
            try:
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0
                )
            except asyncio.TimeoutError:
                # Send keepalive ping
                await websocket.send_json({"type": "ping"})
                continue

            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON received")
                continue

            msg_type = message.get("type", "")

            if msg_type == "pong":
                continue
            
            if msg_type == "config":
                # Client sending configuration
                logger.info(f"Client config: {message}")
                continue

            if msg_type == "frame":
                frame_count += 1
                
                # Skip frames for performance
                if frame_count % config.frame_skip != 0:
                    continue

                # Process frame
                result = await process_frame(
                    message, body_analyzer
                )
                
                # Send result back
                await websocket.send_json(result.to_dict())
                
                state.total_frames_processed += 1
                # Running average
                alpha = 0.1
                state.avg_processing_time_ms = (
                    alpha * result.processing_time_ms +
                    (1 - alpha) * state.avg_processing_time_ms
                )

    except WebSocketDisconnect:
        logger.info(f"📱 Client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        state.connected_clients -= 1
        logger.info(f"Clients remaining: {state.connected_clients}")


async def process_frame(message: dict, body_analyzer: BodyLanguageAnalyzer):
    """Process a single frame (image + audio) from the app."""
    from brain.behavior_engine import AnalysisResult
    
    start = time.time()
    
    # Decode image
    image = None
    image_b64 = message.get("image", "")
    if image_b64:
        try:
            img_bytes = base64.b64decode(image_b64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        except Exception as e:
            logger.warning(f"Failed to decode image: {e}")

    # Decode audio
    audio = None
    audio_b64 = message.get("audio", "")
    if audio_b64:
        try:
            audio_bytes = base64.b64decode(audio_b64)
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as e:
            logger.warning(f"Failed to decode audio: {e}")

    # ── Vision Pipeline ──
    body_signals = None
    if image is not None and state.detector is not None:
        detections = state.detector.detect(image)
        if detections:
            best_detection = max(detections, key=lambda d: d.confidence)
            logger.info(f"Vision Detection: {len(detections)} objects found. Best confidence: {best_detection.confidence:.2f}")
            if best_detection.keypoints:
                body_signals = body_analyzer.analyze(best_detection.keypoints)

    # ── Audio Pipeline ──
    vocal_result = None
    if audio is not None and state.sound_classifier is not None:
        vocal_result = state.sound_classifier.classify(audio)

    # ── Behavior Engine ──
    if state.behavior_engine:
        result = state.behavior_engine.analyze(
            body_signals=body_signals,
            vocal_result=vocal_result,
            image_base64=image_b64 if image_b64 else None
        )
    else:
        result = AnalysisResult(
            dog_detected=body_signals is not None,
            interpretation="Server not fully initialized"
        )

    result.processing_time_ms = (time.time() - start) * 1000
    logger.info(f"Processed frame: dog_detected={result.dog_detected}, time={result.processing_time_ms:.1f}ms, interpretation='{result.interpretation}'")
    return result


# ─── Entry Point ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"""
    ============================================================
    Dog Talk AI Server Online
    ============================================================
    Listening on: ws://0.0.0.0:{config.port}/ws
    Ready to process frames from the Android app!
    ============================================================
    """)
    
    uvicorn.run(
        "main:app",
        host=config.host,
        port=config.port,
        reload=False,
        log_level="info",
        ws_ping_interval=20,
        ws_ping_timeout=20,
    )
