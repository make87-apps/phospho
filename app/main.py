import logging
import time
from typing import Optional

import cv2
import httpx
import make87
import numpy as np
import uvicorn
import zenoh
from make87.encodings import ProtobufEncoder
from make87.interfaces.zenoh import ZenohInterface
from make87_messages.core.empty_pb2 import Empty
from make87_messages.image.compressed.image_jpeg_pb2 import ImageJPEG
from phosphobot.am import Pi0

logger = logging.getLogger(__name__)

PHOSPHO_SERVER_PORT = 8473

async def run_model():
    config = make87.config.load_config_from_env()
    pi_interface = config.interfaces.get("pi0-interface")
    if not pi_interface:
        logger.warning("No pi0-interface found in configuration.")
        return

    pi_client = pi_interface.clients.get("pi0")
    if not pi_client:
        logger.warning("No pi0 client found in configuration.")
        return

    zenoh_interface = ZenohInterface(name="zenoh-client", make87_config=config)
    action_publisher = zenoh_interface.get_publisher(name="AGENT_LOGS")
    agent_chat_provider = zenoh_interface.get_provider(name="AGENT_CHAT")
    wrist_cam = zenoh_interface.get_requester(name="GET_WRIST_IMAGE")
    context_cam = zenoh_interface.get_requester(name="GET_CONTEXT_IMAGE")

    PHOSPHOBOT_API_URL = f"http://localhost:{PHOSPHO_SERVER_PORT}"
    model = Pi0(server_url=pi_client.vpn_ip, server_port=pi_client.vpn_port)

    while True:
        try:
            prompt = agent_chat_provider.recv()
        except Exception as e:
            logger.error(f"Error receiving prompt: {e}")
            time.sleep(1)
            continue

        try:
            wrist_image = get_rgb_from_requester(requester=wrist_cam)
            context_image = get_rgb_from_requester(requester=context_cam)
            if context_image is None or wrist_image is None:
                missing = []
                if context_image is None:
                    missing.append("context")
                if wrist_image is None:
                    missing.append("wrist")
                msg = f"Missing images: {', '.join(missing)}"
                logger.warning(msg)
                prompt.reply(key_expr=prompt.key_expr, payload=msg.encode("utf-8"))
                continue
        except Exception as e:
            logger.error(f"Error retrieving images: {e}")
            prompt.reply(key_expr=prompt.key_expr, payload=b"Image acquisition failed.")
            continue

        try:
            response = httpx.post(f"{PHOSPHOBOT_API_URL}/joints/read", timeout=2)
            response.raise_for_status()
            state = response.json()
        except Exception as e:
            logger.error(f"Error reading joints: {e}")
            prompt.reply(key_expr=prompt.key_expr, payload=b"Failed to read joint state.")
            continue

        try:
            inputs = {
                "state": np.array(state["angles_rad"]),
                "images": np.array([context_image, wrist_image]),
                "prompt": prompt,
            }
            actions = model(inputs)
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            prompt.reply(key_expr=prompt.key_expr, payload=b"Model failed to generate action.")
            continue

        for action in actions:
            try:
                action_publisher.put(payload=str(action).encode("utf-8"))
            except Exception as e:
                logger.error(f"Error publishing action: {e}")

            try:
                httpx.post(
                    f"{PHOSPHOBOT_API_URL}/joints/write",
                    json={"angles": action.tolist()},
                    timeout=2,
                )
            except Exception as e:
                logger.error(f"Error writing joints: {e}")
                prompt.reply(key_expr=prompt.key_expr, payload=b"Failed to send joint action.")
                continue

            time.sleep(1 / 30)

        prompt.reply(key_expr=prompt.key_expr, payload=b"Action executed successfully.")

def get_rgb_from_requester(requester: zenoh.Querier) -> Optional[np.ndarray]:
    response = requester.get(payload=ProtobufEncoder(message_type=Empty).encode(Empty()))
    for r in response:
        if r.ok is not None:
            jpeg_bytes = ProtobufEncoder(message_type=ImageJPEG).decode(r.ok.payload.to_bytes())
            ndarray = np.frombuffer(jpeg_bytes.data, dtype=np.uint8)
            image = cv2.imdecode(ndarray, cv2.IMREAD_COLOR)
            # resize to (240, 320)
            ret_image = cv2.resize(image, (320, 240))
            return ret_image
    return None



async def run_app():
    from phosphobot.configs import config as phospho_config
    phospho_config.PORT = PHOSPHO_SERVER_PORT  # <- force app's config to match

    config = uvicorn.Config("phosphobot.app:app", host="0.0.0.0", port=PHOSPHO_SERVER_PORT, reload=False)
    server = uvicorn.Server(config)
    await server.serve()



async def main():
    app_task = asyncio.create_task(run_app())
    model_task = asyncio.create_task(run_model())

    await asyncio.gather(app_task, model_task)


if __name__ == "__main__":
    import asyncio

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Shutting down the server...")
