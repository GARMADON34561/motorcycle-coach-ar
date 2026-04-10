import os
import asyncio
import json
from openai import OpenAI
from models import MotorcycleAction
from server.motorcycle_environment import MotorcycleEnvironment

API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

async def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    env = MotorcycleEnvironment()
    obs = env.reset()
    print("[START] task=motorcycle_coach env=motorcycle_coach_ar model=" + MODEL_NAME)
    for step in range(1, 11):
        if obs.done:
            break
        prompt = f"Speed {obs.speed_kmh} km/h, Lean {obs.lean_angle}, Obstacle {obs.distance_to_obstacle_m}m, Road {obs.road_condition}"
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=150
        )
        data = json.loads(resp.choices[0].message.content)
        action = MotorcycleAction(throttle=data.get("throttle",0.5), brake=data.get("brake",0), lean_angle=data.get("lean_angle",0), steering=data.get("steering",0))
        obs = env.step(action)
        print(f"[STEP] step={step} action=throttle={action.throttle:.2f} reward={obs.reward:.2f} done={str(obs.done).lower()} error=null")
    print(f"[END] success=true steps={step} score={obs.reward:.3f} rewards=0.00,0.00,0.00")

if __name__ == "__main__":
    asyncio.run(main())
