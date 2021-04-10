#include <stdio.h>
#include "cuda/scene.cuh"

int main() {
    Scene main_scene;
    main_scene.create_default_scene();
    main_scene.render();
}