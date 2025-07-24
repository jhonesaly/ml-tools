# ml-tools

Conjunto de ferramentas e estudo do tema: machine learning

## WSL

### Instalação

No powershell use os comandos:

```bash
wsl --install -d Ubuntu
sudo apt update && sudo apt upgrade -y
sudo apt install python3-dev python3-venv build-essential
```

Para abrir novamente o WSL, abra o powershell e use o comando:

```bash
wsl -d Ubuntu
```

No WSL, mude para pasta em que o projeto está (endereço com /mnt/ antes):

```bash
cd /mnt/c/Users/user/Documents/projetos/ml-tools
```

Crie um ambiente virtual e o ative:

```bash
python3 -m venv venv-wsl
source venv-wsl/bin/activate
```