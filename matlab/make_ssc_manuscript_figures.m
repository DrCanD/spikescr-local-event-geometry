function make_ssc_manuscript_figures(dataDir, outDir)
%MAKE_SSC_MANUSCRIPT_FIGURES Publication-ready MATLAB figures for the SSC paper.
% Descriptive naming update. Plotting statements and numerical inputs are unchanged.
%
% Usage
%   make_ssc_manuscript_figures
%   make_ssc_manuscript_figures('/path/to/data','/path/to/output')
%
% Inputs are the four CSV files distributed with this script.
%
% Canonical geometry provenance:
%   Original exhaustive local-neighborhood result archive
%   SHA-256: fc6669b840bd473bb1f4e16755e417c8530b971124a6dde714ef861519fc1248
%   Frozen local-neighborhood protocol SHA-256: ded89cf6e33d61ab4ae6d24a51f920a69e9a9cef0e9ae88741a2f91edabeaa6e
%
% Internal-summary values are the activation-analysis values in the revised manuscript:
% trace-contrast table and adverse matched-activation-replacement table.
%
% Outputs: vector PDF + 600-dpi PNG for Figures 1 to 4.

if nargin < 1 || isempty(dataDir)
    dataDir = fileparts(mfilename('fullpath'));
end
if nargin < 2 || isempty(outDir)
    outDir = fullfile(dataDir,'figures');
end
if ~exist(outDir,'dir')
    mkdir(outDir);
end

% ----------------------------- style -------------------------------------
C.navy       = [0.00 0.20 0.36];
C.blue       = [0.12 0.42 0.68];
C.orange     = [0.82 0.47 0.13];
C.green      = [0.12 0.52 0.30];
C.red        = [0.72 0.20 0.20];
C.purple     = [0.47 0.31 0.62];
C.gray       = [0.64 0.64 0.64];
C.darkgray   = [0.28 0.28 0.28];
C.lightgray  = [0.93 0.93 0.93];
C.verylight  = [0.97 0.98 0.99];

fontName = 'Arial';
fontSize = 8.5;
lineWidth = 1.25;

% ============================ FIGURE 1 ===================================
fig = figure('Color','w','Units','centimeters','Position',[2 2 19.5 9.8]);
ax = axes(fig,'Position',[0.025 0.035 0.95 0.93]);
hold(ax,'on'); axis(ax,[0 1 0 1]); axis(ax,'off');

% Main model / benchmark branch -- enlarged boxes for publication readability
box(ax,[0.025 0.70 0.23 0.19],C.verylight,C.navy,1.2);
text(ax,0.140,0.815,'Frozen SpikeSCR','HorizontalAlignment','center','FontName',fontName,'FontSize',10,'FontWeight','bold','Color',C.navy);
text(ax,0.140,0.755,'seed 312, epoch 282','HorizontalAlignment','center','FontName',fontName,'FontSize',fontSize,'Color',C.darkgray);
arrow(ax,0.255,0.795,0.310,0.795,C.darkgray);

box(ax,[0.310 0.70 0.25 0.19],[1 1 1],C.darkgray,1.0);
text(ax,0.435,0.815,'Official SSC test','HorizontalAlignment','center','FontName',fontName,'FontSize',9.2,'FontWeight','bold');
text(ax,0.435,0.755,'84.6188%  (benchmark only)','HorizontalAlignment','center','FontName',fontName,'FontSize',8.0,'Color',C.darkgray);
text(ax,0.435,0.720,'no perturbation / no tuning','HorizontalAlignment','center','FontName',fontName,'FontSize',7.4,'Color',C.darkgray);

% Validation audit pipeline
arrow(ax,0.140,0.70,0.140,0.61,C.darkgray);
box(ax,[0.025 0.43 0.23 0.17],[1 1 1],C.navy,1.0);
text(ax,0.140,0.535,'100 fixed validation','HorizontalAlignment','center','FontName',fontName,'FontSize',9,'FontWeight','bold');
text(ax,0.140,0.485,'utterances','HorizontalAlignment','center','FontName',fontName,'FontSize',9,'FontWeight','bold');

arrow(ax,0.255,0.515,0.295,0.515,C.darkgray);
box(ax,[0.295 0.385 0.33 0.25],C.verylight,C.blue,1.2);
text(ax,0.460,0.585,'One-count adjacent-bin operator','HorizontalAlignment','center','FontName',fontName,'FontSize',9.3,'FontWeight','bold','Color',C.navy);
text(ax,0.460,0.525,'X(t,f) = X(t,f) - 1','HorizontalAlignment','center','FontName',fontName,'FontSize',8.7,'Interpreter','tex');
text(ax,0.460,0.480,'X(t\pm1,f) = X(t\pm1,f) + 1','HorizontalAlignment','center','FontName',fontName,'FontSize',8.7,'Interpreter','tex');
text(ax,0.460,0.435,'same feature, fixed horizon','HorizontalAlignment','center','FontName',fontName,'FontSize',7.7,'Color',C.darkgray);

arrow(ax,0.625,0.515,0.650,0.515,C.darkgray);
box(ax,[0.650 0.43 0.33 0.17],[1 1 1],C.navy,1.0);
text(ax,0.815,0.535,'Exact singleton enumeration','HorizontalAlignment','center','FontName',fontName,'FontSize',9,'FontWeight','bold');
text(ax,0.815,0.485,'725,070 unique candidates','HorizontalAlignment','center','FontName',fontName,'FontSize',9,'Color',C.navy);

% Outcome and internal-analysis branch
arrow(ax,0.815,0.43,0.815,0.335,C.darkgray);
box(ax,[0.580 0.12 0.22 0.17],[1 1 1],C.darkgray,1.0);
text(ax,0.690,0.230,'Decision map','HorizontalAlignment','center','FontName',fontName,'FontSize',9,'FontWeight','bold');
text(ax,0.690,0.185,'preserved / adverse /','HorizontalAlignment','center','FontName',fontName,'FontSize',7.8);
text(ax,0.690,0.150,'corrective / lateral','HorizontalAlignment','center','FontName',fontName,'FontSize',7.8);

box(ax,[0.810 0.12 0.18 0.17],[1 1 1],C.darkgray,1.0);
text(ax,0.900,0.230,'Internal audit','HorizontalAlignment','center','FontName',fontName,'FontSize',9,'FontWeight','bold');
text(ax,0.900,0.185,'7 boundaries','HorizontalAlignment','center','FontName',fontName,'FontSize',7.8);
text(ax,0.900,0.150,'trace + replacement','HorizontalAlignment','center','FontName',fontName,'FontSize',7.5);

arrow(ax,0.815,0.335,0.690,0.290,C.darkgray);
arrow(ax,0.815,0.335,0.900,0.290,C.darkgray);

box(ax,[0.025 0.12 0.50 0.17],C.lightgray,C.darkgray,0.8);
text(ax,0.275,0.230,'Scope statement','HorizontalAlignment','center','FontName',fontName,'FontSize',8.7,'FontWeight','bold');
text(ax,0.275,0.185,'Complete only for the declared finite neighborhood,','HorizontalAlignment','center','FontName',fontName,'FontSize',7.6);
text(ax,0.275,0.150,'frozen panel, checkpoint, and singleton implementation.','HorizontalAlignment','center','FontName',fontName,'FontSize',7.6);

export_figure(fig,outDir,'Fig1_audit_workflow');
close(fig);

% ============================ FIGURE 2 ===================================
T = readtable(fullfile(dataDir,'fig2_outcome_partition.csv'),'TextType','string');
uniqueN = T.unique_n;
weightedN = T.event_weighted_n;

presU = uniqueN(1); changeU = sum(uniqueN(2:4));
presW = weightedN(1); changeW = sum(weightedN(2:4));

overall = 100 * [presU changeU; presW changeW] ./ [sum(uniqueN); sum(weightedN)];
changed = 100 * [uniqueN(2:4)'; weightedN(2:4)'] ./ [changeU; changeW];

fig = figure('Color','w','Units','centimeters','Position',[2 2 18.0 7.8]);
tl = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');

ax1 = nexttile(tl,1); hold(ax1,'on');
b = bar(ax1,overall,'stacked','BarWidth',0.62);
b(1).FaceColor = C.gray; b(1).EdgeColor = 'none';
b(2).FaceColor = C.red;  b(2).EdgeColor = 'none';
set(ax1,'XTick',1:2,'XTickLabel',{'Unique','Event-weighted'},'FontName',fontName,'FontSize',fontSize,'LineWidth',0.8,'Box','off');
xlabel(ax1,'Candidate accounting','FontName',fontName,'FontSize',fontSize+0.2);
ylabel(ax1,'Share of all candidates (%)','FontName',fontName,'FontSize',fontSize+0.5);
ylim(ax1,[0 100]); grid(ax1,'on'); ax1.YGrid='on'; ax1.XGrid='off';
legend(ax1,{'Class-preserved','Any class change'},'Location','southoutside','Orientation','horizontal','Box','off','FontSize',7.5);
text(ax1,1,overall(1,1)-4,sprintf('%.2f%%',overall(1,1)),'HorizontalAlignment','center','FontName',fontName,'FontSize',7.5,'Color','w','FontWeight','bold');
text(ax1,2,overall(2,1)-4,sprintf('%.2f%%',overall(2,1)),'HorizontalAlignment','center','FontName',fontName,'FontSize',7.5,'Color','w','FontWeight','bold');
text(ax1,1,overall(1,1)+overall(1,2)/2,sprintf('%.2f%%',overall(1,2)),'HorizontalAlignment','center','FontName',fontName,'FontSize',6.8,'Color','w','FontWeight','bold');
text(ax1,2,overall(2,1)+overall(2,2)/2,sprintf('%.2f%%',overall(2,2)),'HorizontalAlignment','center','FontName',fontName,'FontSize',6.8,'Color','w','FontWeight','bold');
text(ax1,0.02,0.97,'A','Units','normalized','FontName',fontName,'FontSize',10,'FontWeight','bold','VerticalAlignment','top');

ax2 = nexttile(tl,2); hold(ax2,'on');
b = bar(ax2,changed,'stacked','BarWidth',0.62);
b(1).FaceColor = C.red;    b(1).EdgeColor='none';
b(2).FaceColor = C.green;  b(2).EdgeColor='none';
b(3).FaceColor = C.blue;   b(3).EdgeColor='none';
set(ax2,'XTick',1:2,'XTickLabel',{'Unique','Event-weighted'},'FontName',fontName,'FontSize',fontSize,'LineWidth',0.8,'Box','off');
xlabel(ax2,'Class-change accounting','FontName',fontName,'FontSize',fontSize+0.2);
ylabel(ax2,'Composition of class changes (%)','FontName',fontName,'FontSize',fontSize+0.5);
ylim(ax2,[0 100]); grid(ax2,'on'); ax2.YGrid='on'; ax2.XGrid='off';
legend(ax2,{'Adverse','Corrective','Lateral'},'Location','southoutside','Orientation','horizontal','Box','off','FontSize',7.5);
for i = 1:2
    base = 0;
    for j = 1:3
        v = changed(i,j);
        if v > 8
            text(ax2,i,base+v/2,sprintf('%.1f%%',v),'HorizontalAlignment','center','FontName',fontName,'FontSize',7.2,'Color','w','FontWeight','bold');
        end
        base = base + v;
    end
end
text(ax2,0.02,0.97,'B','Units','normalized','FontName',fontName,'FontSize',10,'FontWeight','bold','VerticalAlignment','top');

export_figure(fig,outDir,'Fig2_complete_outcome_partition');
close(fig);

% ============================ FIGURE 3 ===================================
S = readtable(fullfile(dataDir,'fig3_source_geometry.csv'),'TextType','string');
cleanCorrectMask = asLogicalMask(S.clean_correct, 'clean_correct');
C0 = S(cleanCorrectMask,:);

fig = figure('Color','w','Units','centimeters','Position',[2 2 18.0 7.8]);
tl = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');

ax1 = nexttile(tl,1); hold(ax1,'on');
zeroMask = C0.adverse_unique == 0;
nonzeroMask = ~zeroMask;
scatter(ax1,C0.clean_margin(zeroMask),100*C0.adverse_rate(zeroMask),22,C.gray,'filled','MarkerFaceAlpha',0.70);
scatter(ax1,C0.clean_margin(nonzeroMask),100*C0.adverse_rate(nonzeroMask),38,C.red,'filled','MarkerEdgeColor','w','LineWidth',0.45);
set(ax1,'FontName',fontName,'FontSize',fontSize,'LineWidth',0.8,'Box','off');
xlabel(ax1,'Clean top-two margin','FontName',fontName,'FontSize',fontSize+0.5);
ylabel(ax1,'Adverse-candidate rate (%)','FontName',fontName,'FontSize',fontSize+0.5);
grid(ax1,'on');
text(ax1,0.98,0.79,'Spearman \rho = -0.5688,  p = 0.00010','Units','normalized','HorizontalAlignment','right','VerticalAlignment','top','FontName',fontName,'FontSize',7.3,'BackgroundColor','w','Margin',1);
text(ax1,0.02,0.97,'A','Units','normalized','FontName',fontName,'FontSize',10,'FontWeight','bold','VerticalAlignment','top');
legend(ax1,{'No adverse neighbor','Adverse-sensitive source'},'Location','northeast','Box','off','FontSize',7.2);

ax2 = nexttile(tl,2); hold(ax2,'on');
counts = sort(C0.adverse_unique,'descend');
counts = counts(counts > 0);
cum = 100*cumsum(counts)/sum(counts);
rank = (1:numel(counts))';
plot(ax2,rank,cum,'-o','Color',C.navy,'MarkerFaceColor',C.navy,'MarkerSize',4.0,'LineWidth',lineWidth);
yline(ax2,100,'-','Color',C.gray,'LineWidth',0.8);
plot(ax2,[2 2],[0 cum(2)],'--','Color',C.gray,'LineWidth',0.8);
plot(ax2,[5 5],[0 cum(5)],'--','Color',C.gray,'LineWidth',0.8);
plot(ax2,[8 8],[0 cum(8)],'--','Color',C.gray,'LineWidth',0.8);
text(ax2,2.20,cum(2)+1.5,sprintf('2 sources: %.2f%%',cum(2)),'HorizontalAlignment','left','FontName',fontName,'FontSize',7.2,'BackgroundColor','w','Margin',0.5);
text(ax2,5.00,cum(5)-5.0,sprintf('5 sources: %.2f%%',cum(5)),'HorizontalAlignment','center','FontName',fontName,'FontSize',7.2,'BackgroundColor','w','Margin',0.5);
text(ax2,8.00,cum(8)-7.0,sprintf('8 sources: %.2f%%',cum(8)),'HorizontalAlignment','center','FontName',fontName,'FontSize',7.2,'BackgroundColor','w','Margin',0.5);
set(ax2,'FontName',fontName,'FontSize',fontSize,'LineWidth',0.8,'Box','off','XTick',1:numel(counts));
xlabel(ax2,'Adverse-sensitive sources, ranked by adverse count','FontName',fontName,'FontSize',fontSize+0.5);
ylabel(ax2,'Cumulative adverse load (%)','FontName',fontName,'FontSize',fontSize+0.5);
xlim(ax2,[1 max(rank)]); ylim(ax2,[0 103]); grid(ax2,'on');
text(ax2,0.02,0.97,'B','Units','normalized','FontName',fontName,'FontSize',10,'FontWeight','bold','VerticalAlignment','top');

export_figure(fig,outDir,'Fig3_source_level_fragility');
close(fig);

% ============================ FIGURE 4 ===================================
R = readtable(fullfile(dataDir,'fig4_trace_contrasts.csv'),'TextType','string');
P = readtable(fullfile(dataDir,'fig4_adverse_patching.csv'),'TextType','string');

fig = figure('Color','w','Units','centimeters','Position',[2 2 18.0 8.3]);
tl = tiledlayout(fig,1,2,'TileSpacing','compact','Padding','compact');

ax1 = nexttile(tl,1); hold(ax1,'on');
x = (1:height(R))';
plot(ax1,x,R.adverse_contrast,'-o','Color',C.red,'MarkerFaceColor',C.red,'MarkerSize',4.5,'LineWidth',lineWidth);
plot(ax1,x,R.corrective_contrast,'-s','Color',C.green,'MarkerFaceColor',C.green,'MarkerSize',4.2,'LineWidth',lineWidth);
plot(ax1,x,R.lateral_contrast,'-^','Color',C.blue,'MarkerFaceColor',C.blue,'MarkerSize',4.5,'LineWidth',lineWidth);
yline(ax1,0,'-','Color',C.gray,'LineWidth',0.8);
set(ax1,'XTick',x,'XTickLabel',cellstr(R.boundary),'XTickLabelRotation',32,'FontName',fontName,'FontSize',fontSize,'LineWidth',0.8,'Box','off');
ylabel(ax1,'Transition - preserved relative L_2 change','Interpreter','tex','FontName',fontName,'FontSize',fontSize+0.5);
grid(ax1,'on');
legend(ax1,{'Adverse','Corrective','Lateral'},'Location','northwest','Box','off','FontSize',7.4);
% mark the one nonsignificant contrast (lateral stem) with an open marker
plot(ax1,1,R.lateral_contrast(1),'^','Color',C.blue,'MarkerFaceColor','w','MarkerSize',6.0,'LineWidth',1.1,'HandleVisibility','off');
text(ax1,1.08,R.lateral_contrast(1)+0.010,'Lateral stem: q = 0.1749','Color',C.blue,'FontName',fontName,'FontSize',6.8,'HorizontalAlignment','left');
text(ax1,0.02,0.97,'A','Units','normalized','FontName',fontName,'FontSize',10,'FontWeight','bold','VerticalAlignment','top');

ax2 = nexttile(tl,2); hold(ax2,'on');
x = (1:height(P))';
lo = P.equal_source_mean_percent - P.ci95_low_percent;
hi = P.ci95_high_percent - P.equal_source_mean_percent;
errorbar(ax2,x,P.equal_source_mean_percent,lo,hi,'o','Color',C.navy,'MarkerFaceColor',C.navy,'MarkerSize',5.0,'LineWidth',1.15,'CapSize',6);
plot(ax2,x,P.candidate_pooled_percent,'s','Color',C.orange,'MarkerFaceColor','w','MarkerSize',6.0,'LineWidth',1.25);
set(ax2,'XTick',x,'XTickLabel',cellstr(P.boundary),'XTickLabelRotation',25,'FontName',fontName,'FontSize',fontSize,'LineWidth',0.8,'Box','off');
ylabel(ax2,'Adverse clean-prediction restoration (%)','FontName',fontName,'FontSize',fontSize+0.5);
ylim(ax2,[0 103]); grid(ax2,'on');
legend(ax2,{'Equal-source mean (95% CI)','Candidate-pooled rate'},'Location','southwest','Box','off','FontSize',7.2);
text(ax2,0.02,0.97,'B','Units','normalized','FontName',fontName,'FontSize',10,'FontWeight','bold','VerticalAlignment','top');

export_figure(fig,outDir,'Fig4_internal_divergence_and_recovery');
close(fig);

fprintf('Figures written to: %s\n',outDir);
end

% =========================== helper functions =============================
function box(ax,pos,faceColor,edgeColor,lineWidth)
rectangle(ax,'Position',pos,'FaceColor',faceColor,'EdgeColor',edgeColor,'LineWidth',lineWidth,'Curvature',0.05);
end

function arrow(ax,x1,y1,x2,y2,color)
quiver(ax,x1,y1,x2-x1,y2-y1,0,'Color',color,'LineWidth',1.0,'MaxHeadSize',0.20);
end

function mask = asLogicalMask(v, variableName)
%ASLOGICALMASK Robustly convert table booleans to a logical mask.
% Accepts logical, numeric 0/1, string/categorical, char, or cell text.
if islogical(v)
    mask = v;
elseif isnumeric(v)
    mask = (v ~= 0);
elseif isstring(v)
    s = lower(strtrim(v));
    mask = ismember(s,["true","1","yes","y"]);
elseif iscategorical(v)
    s = lower(strtrim(string(v)));
    mask = ismember(s,["true","1","yes","y"]);
elseif ischar(v)
    s = lower(strtrim(string(cellstr(v))));
    mask = ismember(s,["true","1","yes","y"]);
elseif iscell(v)
    s = lower(strtrim(string(v)));
    mask = ismember(s,["true","1","yes","y"]);
else
    error('Unsupported data type for %s: %s',variableName,class(v));
end
mask = mask(:);
end

function export_figure(fig,outDir,baseName)
pdfPath = fullfile(outDir,[baseName '.pdf']);
pngPath = fullfile(outDir,[baseName '.png']);
exportgraphics(fig,pdfPath,'ContentType','vector');
exportgraphics(fig,pngPath,'Resolution',600);
end
