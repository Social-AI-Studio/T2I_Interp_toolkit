from omegaconf import OmegaConf
import hydra
@hydra.main(config_path=None, version_base=None)
def main(cfg):
    print("target_heads type:", type(cfg.target_heads))
if __name__ == "__main__":
    main()
