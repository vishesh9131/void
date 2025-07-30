/*---------------------------------------------------------------------------------------------
 *  Copyright (c) Microsoft Corporation. All rights reserved.
 *  Licensed under the MIT License. See License.txt in the project root for license information.
 *--------------------------------------------------------------------------------------------*/

import { ViewPane, IViewPaneOptions } from '../../../browser/parts/views/viewPane.js';
import { IInstantiationService } from '../../../../platform/instantiation/common/instantiation.js';
import { IViewDescriptorService } from '../../../common/views.js';
import { IConfigurationService } from '../../../../platform/configuration/common/configuration.js';
import { IContextKeyService } from '../../../../platform/contextkey/common/contextkey.js';
import { IThemeService } from '../../../../platform/theme/common/themeService.js';
import { IContextMenuService } from '../../../../platform/contextview/browser/contextView.js';
import { IKeybindingService } from '../../../../platform/keybinding/common/keybinding.js';
import { IOpenerService } from '../../../../platform/opener/common/opener.js';
import { ITelemetryService } from '../../../../platform/telemetry/common/telemetry.js';
import { IHoverService } from '../../../../platform/hover/browser/hover.js';
import { Registry } from '../../../../platform/registry/common/platform.js';
import { IViewContainersRegistry, Extensions as ViewContainerExtensions, ViewContainerLocation } from '../../../common/views.js';
import { ViewPaneContainer } from '../../../browser/parts/views/viewPaneContainer.js';
import { SyncDescriptor } from '../../../../platform/instantiation/common/descriptors.js';
import { Codicon } from '../../../../base/common/codicons.js';
import { localize2 } from '../../../../nls.js';
import { IViewsRegistry, Extensions as ViewExtensions } from '../../../common/views.js';
import { ICommandService } from '../../../../platform/commands/common/commands.js';
import { Button } from '../../../../base/browser/ui/button/button.js';
import { defaultButtonStyles } from '../../../../platform/theme/browser/defaultStyles.js';

const ML_TOOLS_VIEW_CONTAINER_ID = 'workbench.view.ml-tools';
const ML_TOOLS_VIEW_ID = 'workbench.view.ml-tools.main';

class MLToolsViewPane extends ViewPane {
	private mlToolsContainer: HTMLElement | undefined;

	constructor(
		options: IViewPaneOptions,
		@IInstantiationService instantiationService: IInstantiationService,
		@IViewDescriptorService viewDescriptorService: IViewDescriptorService,
		@IConfigurationService configurationService: IConfigurationService,
		@IContextKeyService contextKeyService: IContextKeyService,
		@IThemeService themeService: IThemeService,
		@IContextMenuService contextMenuService: IContextMenuService,
		@IKeybindingService keybindingService: IKeybindingService,
		@IOpenerService openerService: IOpenerService,
		@ITelemetryService telemetryService: ITelemetryService,
		@IHoverService hoverService: IHoverService,
		@ICommandService private readonly commandService: ICommandService
	) {
		super(options, keybindingService, contextMenuService, configurationService, contextKeyService, viewDescriptorService, instantiationService, openerService, themeService, hoverService);
	}

	protected override renderBody(parent: HTMLElement): void {
		super.renderBody(parent);
		this.mlToolsContainer = parent;
		this.createMLToolsUI();
	}

	private createMLToolsUI(): void {
		if (!this.mlToolsContainer) return;

		this.mlToolsContainer.style.padding = '16px';
		this.mlToolsContainer.style.overflow = 'auto';

		// Title
		const titleElement = document.createElement('h2');
		titleElement.textContent = 'VS Aware ML Tools';
		titleElement.style.marginTop = '0';
		titleElement.style.marginBottom = '16px';
		titleElement.style.fontSize = '18px';
		titleElement.style.fontWeight = 'bold';
		this.mlToolsContainer.appendChild(titleElement);

		// Description
		const descElement = document.createElement('p');
		descElement.textContent = 'Powerful machine learning tools integrated directly into VS Aware';
		descElement.style.marginBottom = '20px';
		descElement.style.color = 'var(--vscode-descriptionForeground)';
		this.mlToolsContainer.appendChild(descElement);

		// ML Tools sections
		const toolSections = [
			{
				title: 'Core ML Tools',
				tools: [
					{ id: 'vsaware.ml.convertNotebook', name: 'Python to Notebook Converter', desc: 'Convert Python scripts to Jupyter notebooks' },
					{ id: 'vsaware.ml.neuralPlayground', name: 'Neural Network Playground', desc: 'Interactive neural network training' },
					{ id: 'vsaware.ml.datasetVisualizer', name: 'Dataset Visualizer', desc: 'Explore and visualize your datasets' },
					{ id: 'vsaware.ml.quickModel', name: 'Quick Model Builder', desc: 'Generate ML model boilerplate code' },
				]
			},
			{
				title: 'Development Tools',
				tools: [
					{ id: 'vsaware.ml.tensorAnalyzer', name: 'Tensor Shape Analyzer', desc: 'Debug tensor dimensions and shapes' },
					{ id: 'vsaware.ml.experimentTracker', name: 'Experiment Tracker', desc: 'Track ML experiments and results' },
					{ id: 'vsaware.ml.codeChecker', name: 'ML Code Quality Checker', desc: 'Analyze ML code best practices' },
				]
			},
			{
				title: 'Advanced Features',
				tools: [
					{ id: 'vsaware.ml.dataGenerator', name: 'Data Generator', desc: 'Generate synthetic datasets' },
					{ id: 'vsaware.ml.modelComparator', name: 'Model Comparator', desc: 'Compare ML model performance' },
					{ id: 'vsaware.ml.hyperTuner', name: 'Hyperparameter Tuner', desc: 'Optimize model hyperparameters' },
				]
			}
		];

		toolSections.forEach(section => {
			this.createToolSection(section.title, section.tools);
		});
	}

	private createToolSection(title: string, tools: Array<{ id: string; name: string; desc: string }>): void {
		if (!this.mlToolsContainer) return;

		// Section title
		const sectionTitle = document.createElement('h3');
		sectionTitle.textContent = title;
		sectionTitle.style.marginTop = '24px';
		sectionTitle.style.marginBottom = '12px';
		sectionTitle.style.fontSize = '14px';
		sectionTitle.style.fontWeight = '600';
		sectionTitle.style.color = 'var(--vscode-foreground)';
		this.mlToolsContainer.appendChild(sectionTitle);

		// Tools in this section
		tools.forEach(tool => {
			const toolContainer = document.createElement('div');
			toolContainer.style.marginBottom = '8px';
			toolContainer.style.padding = '8px';
			toolContainer.style.border = '1px solid var(--vscode-widget-border)';
			toolContainer.style.borderRadius = '4px';
			toolContainer.style.backgroundColor = 'var(--vscode-editor-background)';

			const toolButton = this._register(new Button(toolContainer, defaultButtonStyles));
			toolButton.label = tool.name;
			toolButton.onDidClick(() => {
				this.commandService.executeCommand(tool.id);
			});

			const toolDesc = document.createElement('div');
			toolDesc.textContent = tool.desc;
			toolDesc.style.fontSize = '12px';
			toolDesc.style.color = 'var(--vscode-descriptionForeground)';
			toolDesc.style.marginTop = '4px';
			toolContainer.appendChild(toolDesc);

			if (this.mlToolsContainer) {
				this.mlToolsContainer.appendChild(toolContainer);
			}
		});
	}
}

// Register ML Tools view container
const viewContainerRegistry = Registry.as<IViewContainersRegistry>(ViewContainerExtensions.ViewContainersRegistry);
const mlToolsContainer = viewContainerRegistry.registerViewContainer({
	id: ML_TOOLS_VIEW_CONTAINER_ID,
	title: localize2('mlToolsContainer', 'ML Tools'),
	ctorDescriptor: new SyncDescriptor(ViewPaneContainer, [ML_TOOLS_VIEW_CONTAINER_ID, {
		mergeViewWithContainerWhenSingleView: true,
	}]),
	hideIfEmpty: false,
	order: 7, // Position after Testing (6)
	icon: Codicon.symbolMethod, // Using a gear icon for tools
}, ViewContainerLocation.Sidebar);

// Register ML Tools view
const viewsRegistry = Registry.as<IViewsRegistry>(ViewExtensions.ViewsRegistry);
viewsRegistry.registerViews([{
	id: ML_TOOLS_VIEW_ID,
	hideByDefault: false,
	name: localize2('mlToolsView', 'ML Tools'),
	ctorDescriptor: new SyncDescriptor(MLToolsViewPane),
	canToggleVisibility: true,
	canMoveView: true,
	weight: 100,
	order: 1,
}], mlToolsContainer);

console.log('VS Aware: ML Tools sidebar registered successfully!');
