plugins {
    alias(libs.plugins.android.library)
}

android {
    namespace = "com.n3ttx.aicore.engine"
    compileSdk = 37

    defaultConfig {
        minSdk = 36
        consumerProguardFiles("consumer-rules.pro")
        ndk {
            // Phones only. x86_64 would double the native build time for emulator support.
            abiFilters += "arm64-v8a"
        }
        externalNativeBuild {
            cmake {
                // Inference must be optimized even in debug builds, otherwise it is unusably slow.
                arguments += "-DCMAKE_BUILD_TYPE=Release"
            }
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }
}

dependencies {
    // ONNX Runtime (MIT, from Maven Central) for image, face and generic ONNX models.
    api(libs.onnxruntime.android)
}
