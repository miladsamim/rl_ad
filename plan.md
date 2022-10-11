# Development plan
## Explore carla 
- :heavy_check_mark: set up platform 
- :heavy_check_mark: asses vehicles, sensors, general environment 

## Scenario building
- :heavy_check_mark: explore a way to set up scenario
- :heavy_check_mark: set up first simple straight driving scenario

## Agent Controller
- explore a way to connnect a control to simulation
- extend python scenarios to handle agents, check where the agent sets up its route gps info, and compare with route approach
    - make route builder
    - adapt scenario_runner, add agent1 parameter
    - add criterias to agent
    - check that all data wanted criteria, input data is available for the agent
- build a first simple controller 

## First evaluation
- setup SAC controller to learn straight driving
- Build way to do imitation learning to build sample buffer for learning 
- Test training loop
- Try to learn straight driving

# Next step
- Evaluate setup
- More complex scenarios 

