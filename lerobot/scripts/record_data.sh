python lerobot/scripts/control_robot.py \
    --robot.type=kinova \
    --control.type=record \
    --control.fps=15 \
    --control.single_task="Pick up the orange and put it in the drawer." \
    --control.repo_id=qbb/pick_and_put_in_drawer_v1 \
    --control.num_episodes=600 \
    --control.display_cameras=false \
    --control.push_to_hub=false \
    --control.episode_time_s=70 \
    --control.resume=true

python lerobot/scripts/visualize_dataset.py \                
    --repo-id qbb/pick_and_put_in_drawer_v1 \
    --episode-index 0