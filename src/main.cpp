#include <stdio.h>
#include <iostream>
#include <SDL2/SDL.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "kernel.cuh"

const int SCREEN_WIDTH = 800;
const int SCREEN_HEIGHT = 600;

int main(int argc, char *argv[]) {
    float offsetX = 0.5;
    float offsetY = 0;
    float resolution = 2.5;

    // https://www.perplexity.ai/search/how-to-setup-a-window-where-i-RplzbE4ZQGuQ3Xue_CxoKw
    if (SDL_Init(SDL_INIT_VIDEO) != 0) {
        fprintf(stderr, "SDL_Init Error: %s\n", SDL_GetError());
        return 1;
    }

    SDL_Window* window = SDL_CreateWindow("Mandelbrot Set", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, SCREEN_WIDTH, SCREEN_HEIGHT, SDL_WINDOW_SHOWN);
    if (window == nullptr) {
        std::cerr << "Window could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return 1;
    }

    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (renderer == nullptr) {
        std::cerr << "Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    SDL_Texture* texture = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGB888, SDL_TEXTUREACCESS_STREAMING, SCREEN_WIDTH, SCREEN_HEIGHT);
    if (texture == nullptr) {
        std::cerr << "Texture could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        SDL_DestroyRenderer(renderer);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    Uint32 pixels[SCREEN_WIDTH * SCREEN_HEIGHT]; 
    call_kernel(pixels, SCREEN_WIDTH, SCREEN_HEIGHT, resolution, offsetX, offsetY);

    SDL_UpdateTexture(texture, nullptr, pixels, SCREEN_WIDTH * sizeof(Uint32));

    bool quit = false;
    SDL_Event e;

    while (!quit) {
        while (SDL_PollEvent(&e) != 0) {
            if (e.type == SDL_QUIT) {
                quit = true;
            }
            else if (e.type == SDL_KEYDOWN) {
                // Select actions based on key press
                switch (e.key.keysym.sym) {
                    case SDLK_UP:
                        printf("Up arrow key pressed.\n");
                        offsetY -= 0.1;
                        break;
                    case SDLK_DOWN:
                        printf("Down arrow key pressed.\n");
                        offsetY += 0.1;
                        break;
                    case SDLK_LEFT:
                        printf("Left arrow key pressed.\n");
                        offsetX -= 0.1;
                        break;
                    case SDLK_RIGHT:
                        printf("Right arrow key pressed.\n");
                        offsetX += 0.1;
                        break;
                    case SDLK_PLUS:
                        printf("Plus key pressed.\n");
                        resolution -= 0.3;
                        break;
                    case SDLK_MINUS:
                        printf("Plus key pressed.\n");
                        resolution += 0.3;
                        break;
                }
                call_kernel(pixels, SCREEN_WIDTH, SCREEN_HEIGHT, resolution, offsetX, offsetY);
                SDL_UpdateTexture(texture, nullptr, pixels, SCREEN_WIDTH * sizeof(Uint32));
            }
        }

        SDL_RenderClear(renderer);
        SDL_RenderCopy(renderer, texture, nullptr, nullptr);
        SDL_RenderPresent(renderer);
    }

    SDL_DestroyTexture(texture);
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}
