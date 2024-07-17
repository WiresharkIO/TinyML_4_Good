/**
  ******************************************************************************
  * @file    accelerometer_motion_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    Wed Jul 17 01:20:15 2024
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2024 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef ACCELEROMETER_MOTION_DATA_PARAMS_H
#define ACCELEROMETER_MOTION_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_ACCELEROMETER_MOTION_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_accelerometer_motion_data_weights_params[1]))
*/

#define AI_ACCELEROMETER_MOTION_DATA_CONFIG               (NULL)


#define AI_ACCELEROMETER_MOTION_DATA_ACTIVATIONS_SIZES \
  { 2168, }
#define AI_ACCELEROMETER_MOTION_DATA_ACTIVATIONS_SIZE     (2168)
#define AI_ACCELEROMETER_MOTION_DATA_ACTIVATIONS_COUNT    (1)
#define AI_ACCELEROMETER_MOTION_DATA_ACTIVATION_1_SIZE    (2168)



#define AI_ACCELEROMETER_MOTION_DATA_WEIGHTS_SIZES \
  { 105764, }
#define AI_ACCELEROMETER_MOTION_DATA_WEIGHTS_SIZE         (105764)
#define AI_ACCELEROMETER_MOTION_DATA_WEIGHTS_COUNT        (1)
#define AI_ACCELEROMETER_MOTION_DATA_WEIGHT_1_SIZE        (105764)



#define AI_ACCELEROMETER_MOTION_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_accelerometer_motion_activations_table[1])

extern ai_handle g_accelerometer_motion_activations_table[1 + 2];



#define AI_ACCELEROMETER_MOTION_DATA_WEIGHTS_TABLE_GET() \
  (&g_accelerometer_motion_weights_table[1])

extern ai_handle g_accelerometer_motion_weights_table[1 + 2];


#endif    /* ACCELEROMETER_MOTION_DATA_PARAMS_H */
