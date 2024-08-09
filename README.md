__TinyAI__

> Design Cycle
<img width="751" alt="image" src="https://github.com/WiresharkIO/TinyAI/assets/14985440/00f48379-0e04-46ff-81f6-7a7e78262eeb">

----------------------



---------------------
> Problems existing

1. ..


> Problems resolved

1. "Not able to find Index" in STM32 Cube IDE during Building project
- Try with Project--> c/c++ Index --> Rebuild

2. Not able to connect to serial terminal like "putty"
- Try with checking:
  1. windows-->device manager--> desired COM port properties and set desired communication rate and other vals.
  2. check/set proper TIMER parameters in IDE configuration window.
  3. if using putty check serial communiation configuration and set it equal to the values set elsewhere.

3. During quantization - InferenceError: [ShapeInferenceError] (op_type:ZipMap, node name: ZipMap): [ShapeInferenceError] type case unsupported for symbolic shape inference. inferred=5
- Refer --> https://github.com/onnx/sklearn-onnx/issues/816

4. Invalid Initializer error with auto-generated code X-CUBE-AI
- Refer --> https://community.st.com/t5/edge-ai/invalid-initializer-error-with-auto-generated-code-x-cube-ai/td-p/104534

5. [STM32F411-Discovery Board] HAL_RCC_OscConfig() always returns HSE_TIMEOUT
- Changing from BYPASS clock source to Crystal/Ceramic Resonator in RCC and then power-cycling(RESET) the board was the solution.

6. https://community.st.com/t5/stm32-mcus-boards-and-hardware/i-could-not-upgrade-st-link-firmware-i-always-get-an-error-using/td-p/220944

---------------------
> __References__

[1] "Convert an XGBoost model into ONNX", ONNX, Available: https://onnx.ai/sklearn-onnx/auto_tutorial/plot_gexternal_xgboost.html

[2] "Model Optimization", ONNX Runtime, Available: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html
