set -x

# Example command
# ```
# ./scripts/run.sh b  "dataset/Abraham Lincoln_01.png" 0.75
# ```

spectral_sensitivity="$1"
path="$2"
blur_radius="$3"


list="$(dirname "${path}")"
list="$(basename "${list}")"

if [ "${spectral_sensitivity}" == "b" ]; then
  FLAGS=(--spectral_sensitivity b --encoder_ckpt checkpoint/encoder/checkpoint_b.pt);
elif [ "${spectral_sensitivity}" == "gb" ]; then
  FLAGS=(--spectral_sensitivity "gb" --encoder_ckpt checkpoint/encoder/checkpoint_gb.pt);
else
  FLAGS=(--spectral_sensitivity "g" --encoder_ckpt checkpoint/encoder/checkpoint_g.pt);
fi

name="${path%.*}"
name="${name##*/}"
echo "${name}"

# TODO: I did l2 or cos for contextual
time python projector.py \
    "${path}"  \
    --gaussian "${blur_radius}" \
    --log_dir "log/"  \
    --results_dir "results/" \
    "${FLAGS[@]}"
