ls -al ~/.ssh              # veja se há id_ed25519 e id_ed25519.pub
ssh-keygen -t ed25519 -C "mbenedicto@gmail.com"   # aceite o caminho padrão (~/.ssh/id_ed25519)
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519

