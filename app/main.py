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
from make87_messages.image.compressed.image_jpeg_pb2 import ImageJPEG
from phosphobot.am import Pi0
from pyparsing import Empty

logger = logging.getLogger(__name__)

PHOSPHO_SERVER_PORT = 8473

async def run_model():
    config = make87.config.load_config_from_env()
    pi_interface = config.interfaces.get("pi0-interface", None)
    if pi_interface is None:
        logger.info(
            "No pi0-interface found in the configuration. "
            "Please ensure that the interface is correctly configured."
        )
        return
    pi_client = pi_interface.clients.get("pi0", None)
    if pi_client is None:
        logger.info(
            "No pi0 client found in the configuration. "
            "Please ensure that the client is correctly configured."
        )
        return


    zenoh_interface = ZenohInterface(name="zenoh-client", make87_config=config)
    action_publisher = zenoh_interface.get_publisher(name="AGENT_LOGS")
    agent_chat_provider = zenoh_interface.get_provider(name="AGENT_CHAT")
    wrist_cam = zenoh_interface.get_requester(name="GET_WRIST_IMAGE")
    context_cam = zenoh_interface.get_requester(name="GET_CONTEXT_IMAGE")

    PHOSPHOBOT_API_URL = f"http://localhost:{PHOSPHO_SERVER_PORT}"


    # Instantiate the model
    model = Pi0(server_url=pi_client.vpn_ip, server_port=pi_client.vpn_port)

    while True:

        try:
            pass
            prompt = agent_chat_provider.recv()


            wrist_image = get_rgb_from_requester(requester=wrist_cam)
            context_image = get_rgb_from_requester(requester=context_cam)
            if not context_image:
                prompt.reply(
                    key_expr=prompt.key_expr,
                    payload="Context image not available".encode("utf-8")
                )
                continue
            if not wrist_image:
                prompt.reply(
                    key_expr=prompt.key_expr,
                    payload="Wrist image not available".encode("utf-8")
                )
                continue

            state = httpx.post(f"{PHOSPHOBOT_API_URL}/joints/read").json()

            inputs = {
                "state": np.array(state["angles_rad"]),
                "images": np.array([context_image, wrist_image]),
                "prompt": prompt,
            }

            # Go through the model
            actions = model(inputs)

            for action in actions:
                try:
                    action_publisher.put(payload=str(action).encode("utf-8"))
                except Exception as e:
                    logger.error(f"Error publishing action: {e}")
                # Send the new joint postion to the robot
                httpx.post(
                    f"{PHOSPHOBOT_API_URL}/joints/write", json={"angles": action.tolist()}
                )
                # Wait to respect frequency control (30 Hz)
                time.sleep(1 / 30)
            prompt.reply(key_expr=prompt.key_expr, payload="Action executed successfully".encode("utf-8"))

        except Exception as e:
            logger.error(f"Error in model loop: {e}")
            time.sleep(1)
            continue

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
    from phosphobot.app import app
    config = uvicorn.Config(app=app, host="0.0.0.0", port=PHOSPHO_SERVER_PORT, reload=False)
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
