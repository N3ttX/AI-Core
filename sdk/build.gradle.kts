plugins {
    alias(libs.plugins.android.library)
}

android {
    namespace = "com.n3ttx.aicore.sdk"
    compileSdk = 36

    defaultConfig {
        minSdk = 36
        consumerProguardFiles("consumer-rules.pro")
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    buildFeatures {
        aidl = true
    }
}

dependencies {
    api(libs.kotlinx.coroutines.android)
}
