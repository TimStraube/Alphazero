#!/bin/bash
set -eo pipefail

# Colored output
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
RED="\033[31m"
RESET="\033[0m"

setup_pyenv(){
    # initialize pyenv in the same way as README recommends
    export PYENV_ROOT="$HOME/.pyenv" && export PATH="$PYENV_ROOT/bin:$PATH" && eval "$(pyenv init -)" 2>/dev/null || true
    pyenv activate alphazero || true
}

echo -e "${BLUE}Starting evaluation: plotting avg episodes from TensorBoard logs...${RESET}"
setup_pyenv

OUT_PATH="eval/results/avg_episodes.png"
echo -e "${YELLOW}Scanning logs directory: logs${RESET}"
echo -e "${YELLOW}Output will be written to: ${OUT_PATH}${RESET}"

if poetry run python3 eval/plot_avg_episodes.py --logdir logs --out "$OUT_PATH" --smooth 0.5; then
    echo -e "${GREEN}Evaluation complete.${RESET}"
    echo -e "${GREEN}Plot saved to:${RESET} ${OUT_PATH}"
    echo -e "${BLUE}Open the file or view in a browser. To start tensorboard run:${RESET}"
    echo -e "  ${YELLOW}poetry run tensorboard --logdir logs --port 6006${RESET}"
else
    echo -e "${RED}Evaluation failed. Check logs directory and ensure tensorboard and matplotlib are installed.${RESET}"
    exit 1
fi
