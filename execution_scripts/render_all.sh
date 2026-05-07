scene_names=("chair" "lego" "materials" "drums")
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"

for scene_name in "${scene_names[@]}"
do
    echo "Rendering scene: $scene_name";
    CUDA_VISIBLE_DEVICES=0 python utils/render.py --scene-type "$scene_name";
done
