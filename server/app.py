import uvicorn
from fastapi import FastAPI
from models import MotorcycleAction
from server.motorcycle_environment import MotorcycleEnvironment

app = FastAPI()
env = MotorcycleEnvironment()

@app.post("/reset")
async def reset():
    obs = env.reset()
    return {"observation": obs.dict(), "reward": obs.reward, "done": obs.done}

@app.post("/step")
async def step(action: MotorcycleAction):
    obs = env.step(action)
    return {"observation": obs.dict(), "reward": obs.reward, "done": obs.done}

@app.get("/")
async def root():
    return {"status": "running"}

def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
