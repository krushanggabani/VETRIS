# from ..config.loader import load_config
# from ..engine.sim import SimulationOrchestrator
# from ..dataio.logger import Logger
# from ..dataio.recorder import Recorder

def cmd_simulate(args):
    print("Simulate command called with args:")
    # cfg = load_config(args.config)
    
#     logger = Logger(cfg)
#     rec = Recorder(cfg, enabled=cfg.get("runtime",{}).get("record_every",0) > 0)
#     sim = SimulationOrchestrator(cfg, logger=logger, recorder=rec, headless=args.headless)
#     sim.reset()
#     steps = int(cfg.get("runtime",{}).get("steps", 1000))
#     for _ in range(steps):
#         sim.step()
#     sim.finish()
#     logger.info("simulate", msg="done", steps=steps)