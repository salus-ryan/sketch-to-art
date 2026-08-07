#!/bin/bash
# Live side-by-side-by-side training monitor (macOS compatible)
# Shows loss curves for all 3 models updating every 5 seconds

BOLD='\033[1m'
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
WHITE='\033[1;37m'
NC='\033[0m'

# macOS-compatible helpers (no grep -P)
extract_loss() { grep "Loss:" "$1" 2>/dev/null | sed 's/.*Loss: //' | sed 's/[^0-9.].*//' | tail -1; }
extract_batch() { grep "Batch " "$1" 2>/dev/null | sed 's/.*Batch //' | sed 's/\/.*//' | tail -1; }
extract_epoch() { grep "Epoch " "$1" 2>/dev/null | sed 's/.*Epoch //' | sed 's/\/.*//' | tail -1; }
extract_total() { grep "Batch " "$1" 2>/dev/null | sed 's/.*Batch [0-9]*\///' | sed 's/,.*//' | tail -1; }
extract_step() { grep "step " "$1" 2>/dev/null | sed 's/.*step //' | sed 's/\/.*//' | tail -1; }
extract_coco_losses() { grep "Batch.*Loss:" "$1" 2>/dev/null | sed 's/.*Loss: //' | sed 's/[^0-9.].*//' | tail -20; }
count_done() { grep -c "done" "$1" 2>/dev/null; }
count_pretrain_done() { grep -c "pre-training done" "$1" 2>/dev/null; }

extract_loss_at_batch() {
    local file=$1 batch=$2
    grep "Batch ${batch}/" "$file" 2>/dev/null | head -1 | sed 's/.*Loss: //' | sed 's/[^0-9.].*//'
}

sparkline() {
    local data="$1" color="$2"
    if [ -z "$data" ]; then printf "  (waiting...)"; return; fi
    local chars=(▁ ▂ ▃ ▄ ▅ ▆ ▇ █)
    local vals=($data)
    local n=${#vals[@]}
    if [ $n -eq 0 ]; then printf "  (waiting...)"; return; fi
    local min=999999999 max=0
    for v in "${vals[@]}"; do
        local vi=$(printf "%.0f" "$v" 2>/dev/null)
        [ "$vi" -lt "$min" ] 2>/dev/null && min=$vi
        [ "$vi" -gt "$max" ] 2>/dev/null && max=$vi
    done
    local range=$((max - min))
    [ $range -eq 0 ] && range=1
    printf "  ${color}"
    for v in "${vals[@]}"; do
        local vi=$(printf "%.0f" "$v" 2>/dev/null)
        local idx=$(( (vi - min) * 7 / range ))
        idx=$((7 - idx))
        printf "%s" "${chars[$idx]}"
    done
    printf "${NC}"
}

while true; do
    clear

    B_LOSS=$(extract_loss /tmp/train_baseline.log)
    B_BATCH=$(extract_batch /tmp/train_baseline.log)
    B_EPOCH=$(extract_epoch /tmp/train_baseline.log)
    B_TOTAL=$(extract_total /tmp/train_baseline.log)
    B_DONE=$(count_done /tmp/train_baseline.log)

    B6_LOSS=$(extract_loss /tmp/train_braille6.log)
    B6_STEP=$(extract_step /tmp/train_braille6.log)
    B6_BATCH=$(extract_batch /tmp/train_braille6.log)
    B6_EPOCH=$(extract_epoch /tmp/train_braille6.log)
    B6_TOTAL=$(extract_total /tmp/train_braille6.log)
    B6_DONE=$(count_done /tmp/train_braille6.log)
    B6_PRETRAIN=$(count_pretrain_done /tmp/train_braille6.log)

    B8_LOSS=$(extract_loss /tmp/train_braille8.log)
    B8_STEP=$(extract_step /tmp/train_braille8.log)
    B8_BATCH=$(extract_batch /tmp/train_braille8.log)
    B8_EPOCH=$(extract_epoch /tmp/train_braille8.log)
    B8_TOTAL=$(extract_total /tmp/train_braille8.log)
    B8_DONE=$(count_done /tmp/train_braille8.log)
    B8_PRETRAIN=$(count_pretrain_done /tmp/train_braille8.log)

    B6S_LOSS=$(extract_loss /tmp/train_braille6s.log)
    B6S_STEP=$(extract_step /tmp/train_braille6s.log)
    B6S_BATCH=$(extract_batch /tmp/train_braille6s.log)
    B6S_EPOCH=$(extract_epoch /tmp/train_braille6s.log)
    B6S_TOTAL=$(extract_total /tmp/train_braille6s.log)
    B6S_DONE=$(count_done /tmp/train_braille6s.log)
    B6S_PRETRAIN=$(count_pretrain_done /tmp/train_braille6s.log)

    B_HIST=$(extract_coco_losses /tmp/train_baseline.log)
    B6_HIST=$(extract_coco_losses /tmp/train_braille6.log)
    B8_HIST=$(extract_coco_losses /tmp/train_braille8.log)
    B6S_HIST=$(extract_coco_losses /tmp/train_braille6s.log)

    B10S_LOSS=$(extract_loss /tmp/train_braille10s.log)
    B10S_STEP=$(extract_step /tmp/train_braille10s.log)
    B10S_BATCH=$(extract_batch /tmp/train_braille10s.log)
    B10S_EPOCH=$(extract_epoch /tmp/train_braille10s.log)
    B10S_TOTAL=$(extract_total /tmp/train_braille10s.log)
    B10S_DONE=$(count_done /tmp/train_braille10s.log)
    B10S_PRETRAIN=$(count_pretrain_done /tmp/train_braille10s.log)
    B10S_HIST=$(extract_coco_losses /tmp/train_braille10s.log)

    # Header
    echo -e "${BOLD}╔═══════════════════════════════════════════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║                            🎨 STYLE TRANSFER TRAINING — LIVE MONITOR                                      ║${NC}"
    echo -e "${BOLD}╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════╣${NC}"
    echo -e "${BOLD}║ ${RED}■ BASELINE${NC}${BOLD}     ${GREEN}■ BR-6${NC}${BOLD}       ${BLUE}■ BR-8${NC}${BOLD}       ${MAGENTA}■ BR-6S${NC}${BOLD}           ${WHITE}■ BR-10S${NC}${BOLD}                       ║${NC}"
    echo -e "${BOLD}║ ${RED}(no pretrain)${NC}${BOLD}  ${GREEN}(2⁶=64)${NC}${BOLD}      ${BLUE}(2⁸=256)${NC}${BOLD}     ${MAGENTA}(3⁶=729,±1)${NC}${BOLD}       ${WHITE}(3¹⁰=59049,±1)${NC}${BOLD}                 ║${NC}"
    echo -e "${BOLD}╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════╣${NC}"

    # Status
    B_STATUS="E${B_EPOCH:-?}/2 B${B_BATCH:-?}/${B_TOTAL:-?}"
    [ "${B_DONE}" -ge 4 ] 2>/dev/null && B_STATUS="✅ COMPLETE"

    if [ "${B6_PRETRAIN}" -ge 1 ] 2>/dev/null; then
        B6_STATUS="E${B6_EPOCH:-?}/2 B${B6_BATCH:-?}/${B6_TOTAL:-?}"
    else
        B6_STATUS="Pre-train ${B6_STEP:-?}/2000"
    fi
    [ "${B6_DONE}" -ge 4 ] 2>/dev/null && B6_STATUS="✅ COMPLETE"

    if [ "${B8_PRETRAIN}" -ge 1 ] 2>/dev/null; then
        B8_STATUS="E${B8_EPOCH:-?}/2 B${B8_BATCH:-?}/${B8_TOTAL:-?}"
    else
        B8_STATUS="Pre-train ${B8_STEP:-?}/2000"
    fi
    [ "${B8_DONE}" -ge 4 ] 2>/dev/null && B8_STATUS="✅ COMPLETE"

    if [ "${B6S_PRETRAIN}" -ge 1 ] 2>/dev/null; then
        B6S_STATUS="E${B6S_EPOCH:-?}/2 B${B6S_BATCH:-?}/${B6S_TOTAL:-?}"
    else
        B6S_STATUS="Pre-train ${B6S_STEP:-?}/2000"
    fi
    [ "${B6S_DONE}" -ge 4 ] 2>/dev/null && B6S_STATUS="✅ COMPLETE"

    if [ "${B10S_PRETRAIN}" -ge 1 ] 2>/dev/null; then
        B10S_STATUS="E${B10S_EPOCH:-?}/2 B${B10S_BATCH:-?}/${B10S_TOTAL:-?}"
    else
        B10S_STATUS="Pre-train ${B10S_STEP:-?}/2000"
    fi
    [ "${B10S_DONE}" -ge 4 ] 2>/dev/null && B10S_STATUS="✅ COMPLETE"

    printf "${BOLD}║${NC} ${RED}%-15s${NC} ${GREEN}%-13s${NC} ${BLUE}%-13s${NC} ${MAGENTA}%-17s${NC} ${WHITE}%-17s${NC}      ${BOLD}║${NC}\n" "$B_STATUS" "$B6_STATUS" "$B8_STATUS" "$B6S_STATUS" "$B10S_STATUS"

    # Loss
    B_L=$(printf "%.0f" "$B_LOSS" 2>/dev/null)
    B6_L=$(printf "%.0f" "$B6_LOSS" 2>/dev/null)
    B8_L=$(printf "%.0f" "$B8_LOSS" 2>/dev/null)
    B6S_L=$(printf "%.0f" "$B6S_LOSS" 2>/dev/null)
    B10S_L=$(printf "%.0f" "$B10S_LOSS" 2>/dev/null)

    printf "${BOLD}║${NC} ${RED}L:%-13s${NC} ${GREEN}L:%-11s${NC} ${BLUE}L:%-11s${NC} ${MAGENTA}L:%-15s${NC} ${WHITE}L:%-15s${NC}      ${BOLD}║${NC}\n" \
        "${B_L:-...}" "${B6_L:-...}" "${B8_L:-...}" "${B6S_L:-...}" "${B10S_L:-...}"

    echo -e "${BOLD}╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════╣${NC}"

    # Sparklines
    echo -e "${BOLD}║${NC}  ${YELLOW}COCO Loss Curves (last 20 checkpoints — taller = better):${NC}                                              ${BOLD}║${NC}"
    printf "${BOLD}║${NC}  ${RED}BASE${NC}  "
    sparkline "$B_HIST" "$RED"
    echo ""
    printf "${BOLD}║${NC}  ${GREEN}BR-6${NC}  "
    sparkline "$B6_HIST" "$GREEN"
    echo ""
    printf "${BOLD}║${NC}  ${BLUE}BR-8${NC}  "
    sparkline "$B8_HIST" "$BLUE"
    echo ""
    printf "${BOLD}║${NC}  ${MAGENTA}BR6S${NC}  "
    sparkline "$B6S_HIST" "$MAGENTA"
    echo ""
    printf "${BOLD}║${NC}  ${WHITE}B10S${NC}  "
    sparkline "$B10S_HIST" "$WHITE"
    echo ""

    echo -e "${BOLD}╠═══════════════════════════════════════════════════════════════════════════════════════════════════════════╣${NC}"

    # Head-to-head
    echo -e "${BOLD}║${NC}  ${YELLOW}HEAD-TO-HEAD (COCO loss at same batch — lower wins):${NC}                                                    ${BOLD}║${NC}"
    printf "${BOLD}║${NC}  %-7s ${RED}%-10s${NC} ${GREEN}%-10s${NC} ${BLUE}%-10s${NC} ${MAGENTA}%-10s${NC} ${WHITE}%-10s${NC}\n" "Batch" "BASE" "BR-6" "BR-8" "BR-6S" "BR-10S"

    for checkpoint in 200 400 600 800 1000 1200 1400 1600 1800; do
        bv=$(extract_loss_at_batch /tmp/train_baseline.log $checkpoint)
        b6v=$(extract_loss_at_batch /tmp/train_braille6.log $checkpoint)
        b8v=$(extract_loss_at_batch /tmp/train_braille8.log $checkpoint)
        b6sv=$(extract_loss_at_batch /tmp/train_braille6s.log $checkpoint)
        b10sv=$(extract_loss_at_batch /tmp/train_braille10s.log $checkpoint)

        if [ -n "$bv" ] || [ -n "$b6v" ] || [ -n "$b8v" ] || [ -n "$b6sv" ] || [ -n "$b10sv" ]; then
            bvi=$(printf "%.0f" "$bv" 2>/dev/null)
            b6vi=$(printf "%.0f" "$b6v" 2>/dev/null)
            b8vi=$(printf "%.0f" "$b8v" 2>/dev/null)
            b6svi=$(printf "%.0f" "$b6sv" 2>/dev/null)
            b10svi=$(printf "%.0f" "$b10sv" 2>/dev/null)

            # Find winner among available values
            winner=""; min=999999999
            [ -n "$bvi" ] && [ "$bvi" -gt 0 ] 2>/dev/null && [ "$bvi" -lt "$min" ] && min=$bvi && winner="BASE"
            [ -n "$b6vi" ] && [ "$b6vi" -gt 0 ] 2>/dev/null && [ "$b6vi" -lt "$min" ] && min=$b6vi && winner="BR-6"
            [ -n "$b8vi" ] && [ "$b8vi" -gt 0 ] 2>/dev/null && [ "$b8vi" -lt "$min" ] && min=$b8vi && winner="BR-8"
            [ -n "$b6svi" ] && [ "$b6svi" -gt 0 ] 2>/dev/null && [ "$b6svi" -lt "$min" ] && min=$b6svi && winner="BR6S"
            [ -n "$b10svi" ] && [ "$b10svi" -gt 0 ] 2>/dev/null && [ "$b10svi" -lt "$min" ] && min=$b10svi && winner="B10S"

            printf "${BOLD}║${NC}  B%-5d ${RED}%-10s${NC} ${GREEN}%-10s${NC} ${BLUE}%-10s${NC} ${MAGENTA}%-10s${NC} ${WHITE}%-10s${NC}" \
                "$checkpoint" "${bvi:-···}" "${b6vi:-···}" "${b8vi:-···}" "${b6svi:-···}" "${b10svi:-···}"
            if [ -n "$winner" ]; then
                printf " ← ${CYAN}${BOLD}%s${NC}" "$winner"
            fi
            echo ""
        fi
    done

    echo -e "${BOLD}╚═══════════════════════════════════════════════════════════════════════════════════════════════════════════╝${NC}"
    echo -e "  ${CYAN}Refreshing every 5s · Ctrl+C to exit${NC}  $(date '+%H:%M:%S')"

    sleep 5
done
