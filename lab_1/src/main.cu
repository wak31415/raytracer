#include <json/json.h>
#include <fstream>
#include <stdio.h>
#include <iostream>
#include "scene.cuh"

int main(int argc, char** argv) {
    std::string scene_path = "../assets/scenes/default.json";
    if(argc > 1) {
        scene_path = std::string(argv[1]);
        std::cout << "Using custom scene located at " << scene_path << std::endl;
    }

    Scene main_scene;
    main_scene.load_scene(scene_path);
    main_scene.render("image.png", 1);
}