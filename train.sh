#!/usr/bin/env bash
# Be permissive about unset variables to avoid failures from environment differences
set -eo pipefail

# Interactive training launcher.
# - Lists agents in src/agents
# - Loads hyperparameters from hyperparameter.json (if present)
# - Starts the selected agent with parameters

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HP_JSON="$ROOT_DIR/hyperparameter.json"

print_header(){
    echo "============================================"
    echo "   RL training contoller"
    echo "============================================"
}

list_agents(){
    # list directories under src/agents and top-level py files
    pushd "$ROOT_DIR" > /dev/null
    agents=()
    if [ -d src/agents ]; then
        while IFS= read -r -d $'\0' d; do
            agents+=("$(basename "$d")")
        done < <(find src/agents -maxdepth 1 -mindepth 1 -type d -print0)
        # also pick up direct python files in src/agents
        while IFS= read -r -d $'\0' f; do
            agents+=("$(basename "${f%.*}")")
        done < <(find src/agents -maxdepth 1 -mindepth 1 -type f -name '*.py' -print0)
    fi
    popd > /dev/null
    # unique preserve order
    declare -A seen=()
    uniq_agents=()
    for a in "${agents[@]}"; do
        if [[ -z "${seen[$a]:-}" ]]; then
            uniq_agents+=("$a")
            seen[$a]=1
        fi
    done
    printf "%s\n" "${uniq_agents[@]}"
}

select_agent(){
    mapfile -t AGENTS < <(list_agents)
    if [ ${#AGENTS[@]} -eq 0 ]; then
        echo "No agents found in src/agents" >&2
        exit 1
    fi
    echo "Available agents:" >&2
    for i in "${!AGENTS[@]}"; do
        printf "  %2d) %s\n" $((i+1)) "${AGENTS[$i]}" >&2
    done
    read -rp "Select agent (number or name): " sel >&2
    if [[ "$sel" =~ ^[0-9]+$ ]]; then
        idx=$((sel-1))
        if [ $idx -ge 0 ] && [ $idx -lt ${#AGENTS[@]} ]; then
            echo "${AGENTS[$idx]}"
            return 0
        fi
    else
        # check name
        for a in "${AGENTS[@]}"; do
            if [ "$a" = "$sel" ]; then
                echo "$a"
                return 0
            fi
        done
    fi
    echo "Invalid selection" >&2
    exit 2
}

load_params_for_agent(){
    local agent_name="$1"
    # if hyperparameter.json exists and is valid JSON, extract object for agent
    if [ -f "$HP_JSON" ] && [ -s "$HP_JSON" ]; then
        # use python to safely extract mapping
        params=$(python3 - <<PY
import json,sys
try:
    j=json.load(open('$HP_JSON'))
except Exception:
    sys.exit(0)
obj=j.get('$agent_name', None)
if not obj:
    sys.exit(0)
args=[]
for k,v in obj.items():
    key='--'+k.replace('_','-')
    if isinstance(v, bool):
        if v:
            args.append(key)
    else:
        args.append(key)
        args.append(str(v))
print(' '.join(args))
PY
)
        echo "$params"
    fi
}

detect_entrypoint(){
    # Map agent name to runnable script path under src/agents
    local agent="$1"
    # prefer src/agents/<agent>/<agent>.py or src/agents/<agent>.py or src/agents/<agent>/main.py
    candidates=(
        "$ROOT_DIR/src/agents/$agent/main.py"
    )
    for c in "${candidates[@]}"; do
        if [ -f "$c" ]; then
            echo "$c"
            return 0
        fi
    done
    # fallback: try any single python file in the agent dir
    if [ -d "$ROOT_DIR/src/agents/$agent" ]; then
        f=$(find "$ROOT_DIR/src/agents/$agent" -maxdepth 1 -type f -name '*.py' | head -n1 || true)
        if [ -n "$f" ]; then
            echo "$f"
            return 0
        fi
    fi
    return 1
}

run_command(){
    local cmd="$1"
    echo "Running: $cmd"
    echo "---- begin output ----"
    # run command and stream output
    eval "$cmd"
    status=$?
    echo "---- end output (exit code: $status) ----"
    return $status
}

setup_pyenv(){
    export PYENV_ROOT="$HOME/.pyenv" && export PATH="$PYENV_ROOT/bin:$PATH" && eval "$(pyenv init -)"
    pyenv activate alphazero || true
}

main(){
    print_header
    setup_pyenv
    agent=$(select_agent)
    echo "Selected agent: $agent"
    entry=$(detect_entrypoint "$agent" || true)
    if [ -z "$entry" ]; then
        echo "Could not detect entrypoint for agent '$agent'" >&2
        exit 3
    fi

    params=$(load_params_for_agent "$agent" || true)
    if [ -n "$params" ]; then
        echo "Using hyperparameters from $HP_JSON: $params"
    else
        echo "No hyperparameters found for '$agent' in $HP_JSON — using defaults or CLI args."
    fi

    # choose python runner (use poetry if available). Use `poetry run python` to match common usage.
    if command -v poetry >/dev/null 2>&1 && [ -f pyproject.toml ]; then
        runner="poetry run python"
    else
        runner="python3"
    fi

    # ensure project `src` is on PYTHONPATH so imports like `envs.battleship` work
    if [ -z "${PYTHONPATH:-}" ]; then
        export PYTHONPATH="$ROOT_DIR/src"
    else
        export PYTHONPATH="$ROOT_DIR/src:$PYTHONPATH"
    fi

    # prefer a repository-relative entrypoint (e.g. src/agents/random/random.py)
    rel_entry="$entry"
    if [[ "$entry" == "$ROOT_DIR"* ]]; then
        rel_entry="${entry#$ROOT_DIR/}"
    fi

    cmd="$runner $rel_entry $params"
    run_command "$cmd"
}

main "$@"
