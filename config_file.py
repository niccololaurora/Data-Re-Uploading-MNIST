import os
import shutil

# Chiedi all'utente i valori per --job-name e --output
job_name = input("Inserisci il nome del job (--job-name): ")
output_log = input("Inserisci il nome del file di output (--output): ")

config_file_path = "job_script.sh"

# Leggi il contenuto del file
with open(config_file_path, "r") as file:
    config_content = file.read()

# Sostituisci i valori appropriati nel contenuto del file
config_content = config_content.replace(
    "#SBATCH --job-name=m_design", f"#SBATCH --job-name={job_name}"
)
config_content = config_content.replace(
    "#SBATCH --output=rtqem.log", f"#SBATCH --output={output_log}"
)

# Crea la directory "result" se non esiste già
result_directory = os.path.join("result", job_name)
os.makedirs(result_directory, exist_ok=True)

# Crea il percorso completo per il nuovo file di configurazione
new_config_file_path = os.path.join(result_directory, "config_file.sh")

# Scrivi il nuovo contenuto nel file
with open(new_config_file_path, "w") as file:
    file.write(config_content)


shutil.copy("main.py", result_directory)
shutil.copy("qclass.py", result_directory)
shutil.copy("init.py", result_directory)

print(
    f"File di configurazione aggiornato con successo. Job name: {job_name}, Output: {output_log}"
)
print(f"Nuovo file salvato in: {new_config_file_path}")
print(f"main.py, qclass.py e init.py copiati nella directory result/{job_name}/")
