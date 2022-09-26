import numpy as np
import carla
import cv2


base_sensor_types = ['sensor.camera.semantic_segmentation']
base_sensor_specification = {
    'fov': str(100.0),
    'image_size_x': str(400.0),
    'image_size_y': str(300.0)
}

top_sensor_types = ['sensor.camera.semantic_segmentation']


top_offsets = {
    'hd1': {'x':0, 'y': 0, 'z':40, 'pitch':-90, 'yaw': 0, 'roll':0},
}

top_callbacks = {
    'hd1': lambda sensor, callback, buffer: sensor.listen(lambda data: callback(data, buffer, 'hd1')),
}



model3_base_offsets = {
    'front': {'x':.4, 'y': 0, 'z':1.5, 'pitch':0, 'yaw': 0, 'roll':0},
    'front_right': {'x':.4, 'y': 0, 'z':1.5, 'pitch':0, 'yaw': 60, 'roll':0},
    'front_left': {'x':.4, 'y': 0, 'z':1.5, 'pitch':0, 'yaw': -60, 'roll':0},
    'side_right': {'x':-.3, 'y': .7, 'z':1.3, 'pitch':0, 'yaw': 90, 'roll':0},
    'side_left': {'x':-.3, 'y': -.7, 'z':1.3, 'pitch':0, 'yaw': -90, 'roll':0},
    'rear': {'x':-2.3, 'y': 0, 'z':1, 'pitch':0, 'yaw': 180, 'roll':0},
}

base_callbacks = {
    'front': lambda sensor, callback, buffer: sensor.listen(lambda data: callback(data, buffer, 'front')),
    'front_right': lambda sensor, callback, buffer: sensor.listen(lambda data: callback(data, buffer, 'front_right')),
    'front_left': lambda sensor, callback, buffer: sensor.listen(lambda data: callback(data, buffer, 'front_left')),
    'side_right': lambda sensor, callback, buffer: sensor.listen(lambda data: callback(data, buffer, 'side_right')),
    'side_left': lambda sensor, callback, buffer: sensor.listen(lambda data: callback(data, buffer, 'side_left')),
    'rear': lambda sensor, callback, buffer: sensor.listen(lambda data: callback(data, buffer, 'rear')),
}

class CarEquipment:
    def __init__(self, world, car, base_sensor_types=base_sensor_types, 
                 top_sensor_types=top_sensor_types):
        self.ego_vehicle = car
        # self.top_sensors = self.__add_top_sensors()
        # self.base_sensors = self.__add_base_sensors()
        self.sensor_data = {}
        self.base_sensor_specification = base_sensor_specification
        self.base_offsets = model3_base_offsets
        self.world = world 
        self.bp_lib = world.get_blueprint_library()

    def toggle_sensors(self, sensor_class, action, callback, listen_d):
        assert sensor_class in ['top', 'base']
        assert action in ['start', 'stop']
        sensors = self.top_sensors if sensor_class == 'top' else self.base_sensors
        if action == 'start':
            for name, sensor in sensors.items(): 
                if not sensor.is_alive:
                    print('sensor is destroyed')
                else: 
                    if not sensor.is_listening:
                        listen_d[name](sensor, callback, self.sensor_data)
        else:
            for sensor in sensors.values():
                sensor.stop()
        
    def build_sensors(self, sensor_class, sensor_type, specification, offsets):
        assert sensor_class in ['top', 'base']
        if sensor_class == 'top':
            self.top_sensors = self.__add_base_sensors(sensor_type, specification, offsets)
        else:
            self.base_sensors = self.__add_base_sensors(sensor_type, specification, offsets)

    def destroy_sensors(self, sensor_class, sensor_type, specification, offsets):
        assert sensor_class in ['top', 'base']
        sensors = self.top_sensors if sensor_class == 'top' else 'base'
        for name, sensor in sensors.items(): 
            sensor.destroy()

    def display_sensors(self, name, layout_func, args):
        cv2.namedWindow(name, cv2.WINDOW_AUTOSIZE)
        while True:
            # # Imshow renders sensor data to display
            layout = layout_func(args)
            cv2.imshow(name, layout)
            
            # Quit if user presses 'q'
            if cv2.waitKey(1) == ord('q'):
                break

        # Close OpenCV window when finished
        cv2.destroyAllWindows()

    def __add_base_sensors(self, sensor_type, specification, offsets):
        sensors = {}
        sensor_bp = self.bp_lib.find(sensor_type)
        sensor_bp = self.__modify_bp(sensor_bp, specification)
        init_transforms = self.__build_transform(offsets)
        for position, init_transform in init_transforms.items():
            sensors[position] = self.world.spawn_actor(sensor_bp, init_transform, attach_to=self.ego_vehicle)
        return sensors

    def __build_transform(self, offsets):
        return {position: carla.Transform(
                            carla.Location(x=ofs['x'], y=ofs['y'], z=ofs['z']),
                            carla.Rotation(ofs['pitch'], ofs['yaw'], ofs['roll']))
                            for position,ofs in offsets.items()}

    def __modify_bp(self, bp, specification):
        for key, val in specification.items():
            bp.set_attribute(key, val)
        return bp 

    def __add_top_sensors(self ):
        pass

    def sem_callback(self, image, data_dict, key):
        image.convert(carla.ColorConverter.CityScapesPalette)
        data_dict[key] = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))

    def __opt_callback(self, data, data_dict, key):
        image = data.get_color_coded_flow()
        img = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        img[:,:,3] = 255
        data_dict[key] = img
