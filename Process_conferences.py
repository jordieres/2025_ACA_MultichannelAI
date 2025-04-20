from MULTIMODAL.ConferenceProcessor import load_config
import yaml


def main(): 

    config_file = "config.yaml"

    # Cargar y listar configuraciones disponibles
    with open(config_file, "r") as f:
        configs = yaml.safe_load(f)

    print("Configuraciones disponibles:")
    for name in configs.keys():
        print(f"- {name}")

    # Load the configuration name
    config_name = input("Introduce el nombre de la configuración a utilizar: ").strip()

    if config_name not in configs:
        print(f"Configuración '{config_name}' no encontrada en {config_file}.")
        return

    processor = load_config(config_file, config_name=config_name)
    processor.run()


if __name__ == "__main__":
    main()