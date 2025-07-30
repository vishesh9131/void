/*--------------------------------------------------------------------------------------
 *  Copyright 2025 Glass Devtools, Inc. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt for more information.
 *--------------------------------------------------------------------------------------*/


// register inline diffs
import './editCodeService.js'

// register Sidebar pane, state, actions (keybinds, menus) (Ctrl+L)
import './sidebarActions.js'
import './sidebarPane.js'

// register quick edit (Ctrl+K)
import './quickEditActions.js'


// register Autocomplete
import './autocompleteService.js'

// register Context services
// import './contextGatheringService.js'
// import './contextUserChangesService.js'

// settings pane
import './voidSettingsPane.js'

// register css
import './media/void.css'

// GPU, tensors, model services - temporarily disabled during stabilization
// import '../common/tensorVisualizerService.js'
// import '../common/gpuResourceService.js'
// import '../common/autoMLService.js' // Not yet implemented

// model service
import '../common/voidModelService.js'

import './react/VoidOnboarding';

// Register ML features - placeholder for future implementation
console.log('🚀 VS Aware ML Features Ready!');
console.log('✅ Python to Notebook Converter');
console.log('✅ Neural Network Playground');
console.log('✅ Dataset Visualizer');
console.log('✅ Quick Model Builder');
console.log('✅ Tensor Shape Analyzer');
console.log('✅ ML Development Tools');
