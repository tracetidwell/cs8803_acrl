Installation steps:
```
git clone https://github.com/mksmsrkn/masac

conda create --name masac python=3.7
pip install requirements.txt

git clone https://github.com/openai/multiagent-competition
cd multiagent-competition/gym_compete
pip install -e .
```

Make sure MuJoCo envs work
```
# Open
python
# then
import gym
import gym_compete
```
