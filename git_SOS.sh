#!/usr/bin/env bash
set -Eeuo pipefail

LOG="git_diag_$(date +%F_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
trap 'echo "[ERROR] Falha na linha $LINENO (exit $?)" >&2' ERR

section() { echo -e "\n=== $* ==="; }
info()    { echo "[INFO] $*"; }
warn()    { echo "[WARN] $*" >&2; }
err()     { echo "[ERRO] $*" >&2; }

section "Verificando se está em um repositório Git"
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  err "Este diretório não é um repositório Git."
  exit 2
fi
ROOT="$(git rev-parse --show-toplevel)"
info "Repo raiz: $ROOT"
cd "$ROOT"

section "Status geral"
git --version
git status
echo

section "Remotos configurados"
if ! git remote -v; then
  err "Nenhum remoto configurado."
fi

section "Branch atual e upstream"
BRANCH="$(git rev-parse --abbrev-ref HEAD || echo 'DESCONHECIDO')"
if [[ "$BRANCH" == "HEAD" ]]; then
  warn "HEAD destacado (detached). Defina/retorne a uma branch."
fi
git branch -vv || true

section "Detalhes do remoto origin"
if git remote get-url origin >/dev/null 2>&1; then
  ORIGIN_URL="$(git remote get-url origin)"
  info "origin = $ORIGIN_URL"
  echo

  section "Teste de acesso ao remoto (ls-remote)"
  if git ls-remote origin >/dev/null 2>&1; then
    info "Conseguiu acessar o remoto."
    REMOTE_OK=1
  else
    REMOTE_OK=0
    RERR="$(git ls-remote origin 2>&1 || true)"
    err "Falha ao acessar o remoto:"
    echo "$RERR"

    # Heurísticas de causa
    if grep -qi "Permission denied (publickey)" <<<"$RERR"; then
      warn "Possível problema de SSH key. Testando 'ssh -T git@github.com'..."
      ssh -T -o BatchMode=yes git@github.com || warn "Falha no SSH para GitHub (verifique chave e SSO)."
    fi
    if grep -qi "Repository not found" <<<"$RERR"; then
      warn "O repositório pode não existir ou você não tem permissão."
    fi
    if grep -qi "The requested URL returned error: 403" <<<"$RERR"; then
      warn "403: token/credencial insuficiente (HTTPS). Use PAT com escopo 'repo'."
    fi
  fi
else
  err "Remoto 'origin' não existe. Configure com:
  git remote add origin <URL-DO-REPO>"
  ORIGIN_URL=""
fi

section "Fetch dry-run (não altera nada)"
git fetch --dry-run origin || warn "Falha no fetch (talvez remoto ausente ou sem permissão)."

section "Ahead/Behind (se upstream existir)"
UPSTREAM=""
if git rev-parse --abbrev-ref --symbolic-full-name "@{u}" >/dev/null 2>&1; then
  UPSTREAM="$(git rev-parse --abbrev-ref --symbolic-full-name "@{u}")"
  info "Upstream de $BRANCH = $UPSTREAM"
  set +e
  AHEAD_BEHIND="$(git rev-list --left-right --count "$UPSTREAM"..."$BRANCH" 2>/dev/null)"
  rc=$?
  set -e
  if [[ $rc -eq 0 ]]; then
    LEFT="${AHEAD_BEHIND%%	*}"
    RIGHT="${AHEAD_BEHIND##*	}"
    info "Behind (local atrás do remoto): $LEFT | Ahead (local à frente): $RIGHT"
  else
    warn "Não foi possível calcular ahead/behind."
  fi
else
  warn "Branch local sem upstream. Defina com: git push -u origin $BRANCH"
fi

section "Push dry-run (simula envio)"
set +e
git push --dry-run origin "$BRANCH" 2>push_err.txt
PUSH_RC=$?
set -e
if [[ $PUSH_RC -eq 0 ]]; then
  info "Push dry-run OK."
else
  err "Push dry-run falhou:"
  cat push_err.txt
  if grep -qi "non-fast-forward" push_err.txt; then
    warn "Rejeitado por não fast-forward. Sugestão:
      git fetch origin
      git pull --rebase origin $BRANCH
      # resolva conflitos se houver, depois:
      git push origin $BRANCH"
  fi
  if grep -qi "protected branch" push_err.txt; then
    warn "Branch protegida no remoto. Abra PR a partir de uma branch de feature."
  fi
  if grep -qi "size .* exceeds GitHub's file size limit" push_err.txt; then
    warn "Arquivo gigante no histórico (>100MB). Use Git LFS ou limpe o histórico (filter-repo/BFG)."
  fi
fi
rm -f push_err.txt

section "Arquivos grandes no diretório de trabalho (>100MB)"
# Ajuda a antecipar bloqueio de push por arquivo grande
FOUND=0
while IFS= read -r -d '' f; do
  sz=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f")
  echo "$sz bytes  $f"
  FOUND=1
done < <(git ls-files -z | xargs -0 -I{} bash -c 'if [ -f "{}" ]; then s=$(stat -c%s "{}" 2>/dev/null || stat -f%z "{}"); if [ "$s" -gt 104857600 ]; then printf "%s\0" "{}"; fi; fi')
if [[ $FOUND -eq 0 ]]; then
  info "Nenhum arquivo >100MB rastreado encontrado."
fi

section "Resumo de possíveis causas (heurísticas)"
if [[ "${REMOTE_OK:-0}" -ne 1 ]]; then
  warn "- Acesso ao remoto falhou (credenciais/URL/repo inexistente)."
fi
if ! git rev-parse --abbrev-ref --symbolic-full-name "@{u}" >/dev/null 2>&1; then
  warn "- Upstream não definido. Use: git push -u origin $BRANCH"
fi

section "Dicas rápidas (comandos úteis)"
cat <<'TIP'
# Definir upstream e enviar
git push -u origin $(git rev-parse --abbrev-ref HEAD)

# Atualizar URL do origin (HTTPS ou SSH)
git remote set-url origin https://github.com/<user>/<repo>.git
# ou
git remote set-url origin git@github.com:<user>/<repo>.git

# Resolver non-fast-forward
git fetch origin
git pull --rebase origin $(git rev-parse --abbrev-ref HEAD)
git push

# Testar SSH
ssh -T git@github.com

# Push simulado
git push --dry-run origin $(git rev-parse --abbrev-ref HEAD)
TIP

section "Fim"
info "Log salvo em: $LOG"

