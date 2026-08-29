plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.tinymlbenchmark"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.example.tinymlbenchmark"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"

        // Explicitly force 64-bit ARM architecture for physical phone execution
        ndk {
            abiFilters.clear()
            abiFilters.addAll(listOf("arm64-v8a"))
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    // Do NOT compress .tflite files when building the APK
    aaptOptions {
        noCompress("tflite")
    }

    packaging {
        jniLibs {
            useLegacyPackaging = false
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.10.0")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.6.2")

    // TensorFlow Lite Runtime (Supports 16 KB page sizes)
    implementation("org.tensorflow:tensorflow-lite:2.17.0")
    implementation("org.tensorflow:tensorflow-lite-api:2.17.0")

    // Flex Delegate runtime
    implementation("org.tensorflow:tensorflow-lite-select-tf-ops:2.16.1")
}