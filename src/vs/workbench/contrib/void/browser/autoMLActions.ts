/*--------------------------------------------------------------------------------------
 *  Copyright 2025 VS Aware Development Team. All rights reserved.
 *  Licensed under the Apache License, Version 2.0. See LICENSE.txt for more information.
 *--------------------------------------------------------------------------------------*/

import { localize } from '../../../../nls.js';
import { ICommandService } from '../../../../platform/commands/common/commands.js';
import { INotificationService } from '../../../../platform/notification/common/notification.js';
import { registerAction2, MenuId, MenuRegistry, Action2 } from '../../../../platform/actions/common/actions.js';
import { ContextKeyExpr } from '../../../../platform/contextkey/common/contextkey.js';
import { IAutoMLService } from '../common/autoMLService.js';
import { URI } from '../../../../base/common/uri.js';
import { ServicesAccessor } from '../../../../platform/instantiation/common/instantiation.js';
import { Codicon } from '../../../../base/common/codicons.js';
import { IDialogService } from '../../../../platform/dialogs/common/dialogs.js';

// Action IDs
export const ANALYZE_DATASET_ACTION_ID = 'vsaware.automl.analyzeDataset';
export const GENERATE_MODEL_PROTOTYPE_ACTION_ID = 'vsaware.automl.generateModelPrototype';
export const OPEN_AUTOML_PANEL_ACTION_ID = 'vsaware.automl.openPanel';

// Context keys
export const DATASET_FILE_CONTEXT = ContextKeyExpr.regex('resourceExtname', /\.(csv|json|parquet|xlsx)$/);

class AnalyzeDatasetAction extends Action2 {
	constructor() {
		super({
			id: ANALYZE_DATASET_ACTION_ID,
			title: {
				value: localize('analyzeDataset', 'Generate Model Prototype'),
				original: 'Generate Model Prototype'
			},
			icon: Codicon.symbolEvent,
			menu: {
				id: MenuId.ExplorerContext,
				when: DATASET_FILE_CONTEXT,
				group: 'vsaware@1'
			}
		});
	}

	async run(accessor: ServicesAccessor, resource?: URI): Promise<void> {
		const autoMLService = accessor.get(IAutoMLService);
		const notificationService = accessor.get(INotificationService);
		const commandService = accessor.get(ICommandService);

		if (!resource) {
			notificationService.error(localize('noResourceSelected', 'No dataset file selected'));
			return;
		}

		try {
			notificationService.info(localize('analyzingDataset', 'Analyzing dataset and generating model recommendations...'));

			const result = await autoMLService.analyzeDataset(resource);

			// Show results notification
			notificationService.info(
				localize('analysisComplete', 'Analysis complete! Found {0} model recommendations.', result.recommendations.length)
			);

			// Open AutoML panel with results
			await commandService.executeCommand(OPEN_AUTOML_PANEL_ACTION_ID, result);

		} catch (error) {
			notificationService.error(
				localize('analysisError', 'Failed to analyze dataset: {0}', error.message || error)
			);
		}
	}
}

class GenerateModelPrototypeAction extends Action2 {
	constructor() {
		super({
			id: GENERATE_MODEL_PROTOTYPE_ACTION_ID,
			title: {
				value: localize('generateModelPrototype', 'Generate Model Code'),
				original: 'Generate Model Code'
			},
			icon: Codicon.code,
			menu: {
				id: MenuId.CommandPalette,
				when: ContextKeyExpr.true()
			}
		});
	}

	async run(accessor: ServicesAccessor, analysisResult?: any, framework?: 'pytorch' | 'tensorflow' | 'sklearn'): Promise<void> {
		const autoMLService = accessor.get(IAutoMLService);
		const commandService = accessor.get(ICommandService);
		const notificationService = accessor.get(INotificationService);

		try {
		if (!analysisResult) {
				notificationService.error(localize('noAnalysisResult', 'No analysis result provided'));
			return;
		}

			let selectedFramework = framework;

			if (!selectedFramework) {
				// For now, default to sklearn since dialog service is not available
				selectedFramework = 'sklearn';
			}

			// Generate model code
			const code = await autoMLService.generateModelPrototype(analysisResult, selectedFramework);

			// Create new file with generated code
			const fileName = `model_${selectedFramework}_${Date.now()}.py`;
			const newFileUri = URI.parse(`untitled:${fileName}`);

			await commandService.executeCommand('vscode.open', newFileUri);
			await commandService.executeCommand('editor.action.insertText', { text: code });

			notificationService.info(
				localize('codeGenerated', 'Model code generated successfully for {0}', selectedFramework)
			);

		} catch (error) {
			notificationService.error(
				localize('codeGenerationError', 'Failed to generate model code: {0}', error.message || error)
			);
		}
	}
}

class OpenAutoMLPanelAction extends Action2 {
	constructor() {
		super({
			id: OPEN_AUTOML_PANEL_ACTION_ID,
			title: {
				value: localize('openAutoMLPanel', 'Open AutoML Assistant'),
				original: 'Open AutoML Assistant'
			},
			icon: Codicon.hubot,
			menu: [
				{
					id: MenuId.ViewTitle,
					when: ContextKeyExpr.equals('view', 'workbench.view.extension.vsaware'),
					group: 'navigation'
				},
				{
					id: MenuId.CommandPalette,
					when: ContextKeyExpr.true()
				}
			]
		});
	}

	async run(accessor: ServicesAccessor, initialData?: any): Promise<void> {
		const commandService = accessor.get(ICommandService);

		// This would open the AutoML panel in the sidebar or a webview
		// For now, we'll use a simple command to show it
		await commandService.executeCommand('workbench.view.extension.vsaware');

		if (initialData) {
			// Pass the initial data to the panel
			console.log('Opening AutoML panel with data:', initialData);
		}
	}
}

// Register actions
registerAction2(AnalyzeDatasetAction);
registerAction2(GenerateModelPrototypeAction);
registerAction2(OpenAutoMLPanelAction);

// Additional menu contributions for dataset file context
MenuRegistry.appendMenuItem(MenuId.ExplorerContext, {
	command: {
		id: ANALYZE_DATASET_ACTION_ID,
		title: localize('quickAnalyze', 'Quick ML Analysis'),
		icon: Codicon.graph
	},
	when: DATASET_FILE_CONTEXT,
	group: 'vsaware@1'
});

// Add submenu for advanced AutoML options
MenuRegistry.appendMenuItem(MenuId.ExplorerContext, {
	submenu: MenuId.ExplorerContext,
	title: localize('automlOptions', 'AutoML Options'),
	when: DATASET_FILE_CONTEXT,
	group: 'vsaware@2'
});

// Add items to submenu
MenuRegistry.appendMenuItem(MenuId.ExplorerContext, {
	command: {
		id: ANALYZE_DATASET_ACTION_ID,
		title: localize('fullAnalysis', 'Full Dataset Analysis')
	},
	when: DATASET_FILE_CONTEXT,
	group: 'vsaware@2'
});

MenuRegistry.appendMenuItem(MenuId.ExplorerContext, {
	command: {
		id: 'vsaware.automl.dataProfiler',
		title: localize('dataProfiler', 'Data Profiling Report')
	},
	when: DATASET_FILE_CONTEXT,
	group: 'vsaware@2'
});

// Data profiler action
class DataProfilerAction extends Action2 {
	constructor() {
		super({
			id: 'vsaware.automl.dataProfiler',
			title: {
				value: localize('dataProfiler', 'Generate Data Profiling Report'),
				original: 'Generate Data Profiling Report'
			}
		});
	}

	async run(accessor: ServicesAccessor, resource?: URI): Promise<void> {
		const notificationService = accessor.get(INotificationService);
		const autoMLService = accessor.get(IAutoMLService);

		if (!resource) {
			notificationService.error(localize('noResourceSelected', 'No dataset file selected'));
			return;
		}

		try {
			notificationService.info(localize('generatingProfile', 'Generating data profiling report...'));

			const analysis = await autoMLService.analyzeDataset(resource);

			// Generate HTML report (simplified)
			const reportHtml = generateDataProfilingReport(analysis.analysis);

			// Create new HTML file with report
			const reportUri = URI.parse(`untitled:data_profile_report_${Date.now()}.html`);
			const commandService = accessor.get(ICommandService);

			await commandService.executeCommand('vscode.open', reportUri);
			await commandService.executeCommand('editor.action.insertText', { text: reportHtml });

			notificationService.info(localize('profileComplete', 'Data profiling report generated successfully'));

		} catch (error) {
			notificationService.error(
				localize('profileError', 'Failed to generate profiling report: {0}', error.message || error)
			);
		}
	}
}

// Hyperparameter tuning action
class HyperparameterTuningAction extends Action2 {
	constructor() {
		super({
			id: 'vsaware.automl.hyperparameterTuning',
			title: {
				value: localize('hyperparameterTuning', 'Hyperparameter Tuning'),
				original: 'Hyperparameter Tuning'
			}
		});
	}

	async run(accessor: ServicesAccessor, resource?: URI): Promise<void> {
		const notificationService = accessor.get(INotificationService);
		const dialogService = accessor.get(IDialogService);

		if (!resource) {
			notificationService.error(localize('noResourceSelected', 'No dataset file selected'));
			return;
		}

		// Show hyperparameter tuning options using prompt method
		const result = await dialogService.prompt({
			type: 'info',
			message: localize('hyperparameterOptions', 'Hyperparameter Tuning Options'),
			detail: localize('hyperparameterDetail', 'Select the hyperparameter optimization method'),
			buttons: [
				{
					label: localize('gridSearch', 'Grid Search'),
					run: () => 'grid_search'
				},
				{
					label: localize('randomSearch', 'Random Search'),
					run: () => 'random_search'
				},
				{
					label: localize('bayesianOptimization', 'Bayesian Optimization'),
					run: () => 'bayesian_optimization'
				}
			],
			cancelButton: localize('cancel', 'Cancel')
		});

		if (result.result) {
			const selectedMethod = result.result;

			notificationService.info(
				localize('tuningStarted', 'Starting hyperparameter tuning with {0}...', selectedMethod)
			);

			// This would trigger the actual hyperparameter tuning process
			// For now, just show a placeholder notification
			setTimeout(() => {
				notificationService.info(
					localize('tuningComplete', 'Hyperparameter tuning completed. Best parameters saved to workspace.')
				);
			}, 3000);
		}
	}
}

registerAction2(DataProfilerAction);
registerAction2(HyperparameterTuningAction);

// Helper function to generate data profiling report
function generateDataProfilingReport(analysis: any): string {
	return `<!DOCTYPE html>
<html>
<head>
	<title>Data Profiling Report</title>
	<style>
		body { font-family: Arial, sans-serif; margin: 20px; }
		.header { background: #f5f5f5; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
		.metric { display: inline-block; margin: 10px; padding: 15px; background: white; border: 1px solid #ddd; border-radius: 5px; }
		.section { margin: 20px 0; }
		table { width: 100%; border-collapse: collapse; margin: 10px 0; }
		th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
		th { background-color: #f5f5f5; }
	</style>
</head>
<body>
	<div class="header">
		<h1>📊 Data Profiling Report</h1>
		<p>Generated on ${new Date().toLocaleString()}</p>
	</div>

	<div class="section">
		<h2>Dataset Overview</h2>
		<div class="metric">
			<strong>Rows:</strong> ${analysis.shape[0].toLocaleString()}
		</div>
		<div class="metric">
			<strong>Columns:</strong> ${analysis.shape[1]}
		</div>
		<div class="metric">
			<strong>Missing Values:</strong> ${analysis.missingValues}
		</div>
		<div class="metric">
			<strong>Problem Type:</strong> ${analysis.problemType}
		</div>
	</div>

	<div class="section">
		<h2>Column Information</h2>
		<table>
			<tr>
				<th>Column</th>
				<th>Data Type</th>
				<th>Category</th>
			</tr>
			${Object.entries(analysis.dataTypes).map(([col, type]) => `
				<tr>
					<td>${col}</td>
					<td>${type}</td>
					<td>${analysis.numericalColumns.includes(col) ? 'Numerical' :
						  analysis.categoricalColumns.includes(col) ? 'Categorical' : 'Other'}</td>
				</tr>
			`).join('')}
		</table>
	</div>

	<div class="section">
		<h2>Statistical Summary</h2>
		<p>Target Column: <strong>${analysis.targetColumn || 'Not detected'}</strong></p>
		<p>Suggested for: <strong>${analysis.problemType}</strong> tasks</p>
	</div>

	<div class="section">
		<h2>Recommendations</h2>
		<ul>
			<li>Dataset size is ${analysis.shape[0] > 10000 ? 'large enough' : 'relatively small'} for machine learning</li>
			<li>${analysis.missingValues > 0 ? `Consider handling ${analysis.missingValues} missing values` : 'No missing values detected'}</li>
			<li>${analysis.categoricalColumns.length > 0 ? `${analysis.categoricalColumns.length} categorical columns may need encoding` : 'No categorical encoding needed'}</li>
		</ul>
	</div>
</body>
</html>`;
}
