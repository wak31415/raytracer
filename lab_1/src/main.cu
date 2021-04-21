#include <json/json.h>
#include <fstream>
#include <stdio.h>
#include <iostream>
#include "scene.cuh"

int main() {
    Scene main_scene;
    main_scene.load_scene("../scene.json");
    // main_scene.create_default_scene();
    main_scene.render();
}