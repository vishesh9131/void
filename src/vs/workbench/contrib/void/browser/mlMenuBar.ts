/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { MenuRegistry, MenuId } from '../../../../platform/actions/common/actions.js';
import { localize } from '../../../../nls.js';

// Create a new MenuId for the Aware Tools submenu
export const MenuId_AwareTools = new MenuId('MenubarAwareToolsMenu');

// Register the main "Aware Tools" menu in the menu bar
MenuRegistry.appendMenuItem(MenuId.MenubarMainMenu, {
	submenu: MenuId_AwareTools,
	title: localize('awareToolsMenu', 'Aware Tools'),
	order: 6, // Position between "Go" and "Terminal"
});

// Add all ML tools to the Aware Tools menu
MenuRegistry.appendMenuItem(MenuId_AwareTools, {
	group: '1_core',
	order: 1,
	command: {
		id: 'vsaware.ml.convertNotebook',
		title: localize('convertNotebook', 'Python to Notebook Converter'),
	}
});

MenuRegistry.appendMenuItem(MenuId_AwareTools, {
	group: '1_core',
	order: 2,
	command: {
		id: 'vsaware.ml.neuralPlayground',
		title: localize('neuralPlayground', 'Neural Network Playground'),
	}
});

MenuRegistry.appendMenuItem(MenuId_AwareTools, {
	group: '1_core',
	order: 3,
	command: {
		id: 'vsaware.ml.datasetVisualizer',
		title: localize('datasetVisualizer', 'Dataset Visualizer'),
	}
});

MenuRegistry.appendMenuItem(MenuId_AwareTools, {
	group: '1_core',
	order: 4,
	command: {
		id: 'vsaware.ml.quickModel',
		title: localize('quickModel', 'Quick Model Builder'),
	}
});

// Separator
MenuRegistry.appendMenuItem(MenuId_AwareTools, {
	group: '2_dev',
	order: 1,
	command: {
		id: 'vsaware.ml.tensorAnalyzer',
		title: localize('tensorAnalyzer', 'Tensor Shape Analyzer'),
	}
});

MenuRegistry.appendMenuItem(MenuId_AwareTools, {
	group: '2_dev',
	order: 2,
	command: {
		id: 'vsaware.ml.experimentTracker',
		title: localize('experimentTracker', 'Experiment Tracker'),
	}
});

MenuRegistry.appendMenuItem(MenuId_AwareTools, {
	group: '2_dev',
	order: 3,
	command: {
		id: 'vsaware.ml.codeChecker',
		title: localize('codeChecker', 'ML Code Quality Checker'),
	}
});

// Separator
MenuRegistry.appendMenuItem(MenuId_AwareTools, {
	group: '3_advanced',
	order: 1,
	command: {
		id: 'vsaware.ml.dataGenerator',
		title: localize('dataGenerator', 'Data Generator'),
	}
});

MenuRegistry.appendMenuItem(MenuId_AwareTools, {
	group: '3_advanced',
	order: 2,
	command: {
		id: 'vsaware.ml.modelComparator',
		title: localize('modelComparator', 'Model Comparator'),
	}
});

MenuRegistry.appendMenuItem(MenuId_AwareTools, {
	group: '3_advanced',
	order: 3,
	command: {
		id: 'vsaware.ml.hyperTuner',
		title: localize('hyperTuner', 'Hyperparameter Tuner'),
	}
});

console.log('VS Aware: Aware Tools menu bar registered successfully!');
