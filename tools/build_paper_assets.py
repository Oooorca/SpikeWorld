"""Numerical plots and tables from preserved result records; preserves draw.io diagrams."""
from pathlib import Path
import json
import hashlib
from statistics import mean
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'results/current'
OUT = ROOT / 'paper/figures'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family':'serif', 'font.serif':['Times New Roman','DejaVu Serif'],
                     'font.size':11, 'pdf.fonttype':42, 'ps.fonttype':42,
                     'axes.spines.top':False, 'axes.spines.right':False})
TEAL, BLUE, RED, PURPLE, GRAY = '#278c88','#3974b9','#d4655b','#6543a4','#6c7482'
records=[]
def read(name):
    p=DATA/name
    records.append({'file':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    return json.loads(p.read_text(encoding='utf-8'))
def save(fig,name):
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.10)
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight',pad_inches=.10)
    fig.savefig(OUT/(name+'.png'),dpi=150,bbox_inches='tight',pad_inches=.10)
    plt.close(fig)
def table(name,caption,label,columns,headers,rows):
    lines=[r'\begin{table}[tbp]',r'\centering\small',r'\caption{'+caption+'}',
           r'\label{'+label+'}',r'\begin{tabular}{@{}'+columns+r'@{}}',r'\toprule',
           ' & '.join(headers)+r' \\',r'\midrule']
    lines += [' & '.join(map(str,row))+r' \\' for row in rows]
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (OUT/(name+'.tex')).write_text('\n'.join(lines)+'\n',encoding='utf-8')

# Figures 1 and 2 are native draw.io sources. Preserve their exported assets.

main=read('deployment_pareto.json'); agg=main['aggregate']
names={'frozen':'Frozen','state_only':'External state','control_adapter':'Adapter (Adam)',
       'full_tuning':'Prediction-path tuning (Adam)','replay':'Replay (Adam)'}
colors={'frozen':GRAY,'state_only':TEAL,'control_adapter':BLUE,'full_tuning':RED,'replay':PURPLE}
order=['frozen','state_only','control_adapter','full_tuning','replay']
rows={r['arm']:r for r in agg['pareto_table']}
fig,axs=plt.subplots(1,2,figsize=(11,4.1),gridspec_kw={'width_ratios':[1.25,1]})
ax=axs[0]; ax.set_xscale('symlog',linthresh=1000)
offsets={'frozen':(10,7),'state_only':(-92,-18),'control_adapter':(7,-20),'full_tuning':(-160,15),'replay':(-20,-38)}
short={'frozen':'Frozen','state_only':'External state\n28.8 KB','control_adapter':'Adapter',
       'full_tuning':'Prediction-path tuning','replay':'Replay'}
for k in order:
    r=rows[k]; xy=(r['mutable_numeric_bytes'],100*r['mean_prediction_gain'])
    ax.scatter(*xy,color=colors[k],s=85,edgecolor='white',linewidth=1,zorder=3)
    ax.annotate(short[k],xy,xytext=offsets[k],textcoords='offset points',fontsize=10,
                arrowprops={'arrowstyle':'-','color':colors[k],'lw':.8})
ax.set(xlabel='Mutable numeric state (bytes; symlog)',ylabel='Shifted prediction gain (%)',ylim=(-2,27))
ax.set_xticks([0,1e4,1e5,1e6],['0',r'$10^4$',r'$10^5$',r'$10^6$'])
ax.grid(axis='y',alpha=.2); ax.set_title('A  Prediction and storage',loc='left',fontweight='bold')
keys=order[1:]; vals=[agg['latency'][k]['median_ms'] for k in keys]
axs[1].barh(range(4),vals,color=[colors[k] for k in keys],height=.56)
axs[1].set_yticks(range(4),['External state','Adapter','Prediction-path tuning','Replay']); axs[1].invert_yaxis()
for i,v in enumerate(vals): axs[1].text(v+3,i,f'{v:.1f}',va='center',fontsize=10)
axs[1].set(xlabel='Median update time (ms)',xlim=(0,210));axs[1].set_title('B  Update latency',loc='left',fontweight='bold')
fig.tight_layout(w_pad=2.2);save(fig,'fig3_tradeoff')
table('table_main','Main six-stage replay comparison. Gains use 126 shifted transitions per run after routing prefixes; all trainable arms use Adam. Mutable state excludes frozen weights and transient activations.','tab:main','lrrrr',
      ['Method','Pred. gain (\%)','Tracking (\%)','Bytes','Update (ms)'],
      [[names[k],f"{100*rows[k]['mean_prediction_gain']:.2f}",f"{100*rows[k]['mean_tracking_gain']:.2f}",f"{rows[k]['mutable_numeric_bytes']:,}", '--' if k=='frozen' else f"{agg['latency'][k]['median_ms']:.2f}"] for k in order])
table('table_foundations','Prediction gain by independent foundation (\%). Each row averages three action seeds.','tab:foundations','lrrrr',
      ['Foundation','External state','Adapter','Prediction-path','Replay'],
      [[r['foundation_seed']]+[f"{100*r['quality'][k]['prediction_improvement_vs_frozen']:.2f}" for k in order[1:]] for r in agg['foundation_rows']])
comp=read('component_ablation.json')['aggregate']
table('table_roles','Component intervention. Full and score-only commands are bitwise equal; matrix contribution is their difference in prediction gain.','tab:roles','lrrrr',
      ['Shift','Full gain (\%)','Scores (\%)','Matrix (points)','Tracking (\%)'],
      [[k.title()]+[f"{100*comp[k][metric]['mean']:.2f}" for metric in ['full_state_prediction','candidate_only_prediction','residual_matrix_prediction_contribution','full_state_action']] for k in ['shear','attenuation']])
control=read('closed_loop_control.json')['aggregate']
joint=read('joint_training.json')['aggregate']
joint_records=read('joint_training.json')['rows']
stage_rows=[]
for label,key in [('Sequential',None),('Action-only shared update','action_only_shared_update'),('Joint shared update','joint_multimodal_action')]:
    summaries=[r['baseline']['summary'] if key is None else r['arms'][key]['evaluation']['summary'] for r in joint_records]
    stage_rows.append([label]+[f"{mean(s[field] for s in summaries)*factor:.4f}" for field,factor in [('action_correct_mse',1),('mean_normalized_next_slice',1),('mean_coupled_semantic_accuracy',100),('binding_mean_class_r1',100)]])
table('table_joint_stages','All stages of the five-action-seed joint-learning study. Sensory MSE is the archived four-group normalized mean; accuracy and retrieval are percentages.','tab:joint_stages','lrrrr',
      ['Stage','Action MSE','Sensory MSE','Semantic (\\%)','Retrieval (\\%)'],stage_rows)
table('table_joint','Joint training relative to the sequential checkpoint. Five action initializations share one sensory foundation. Intervals are paired 95\% bootstrap intervals.','tab:joint','lrr',
      ['Metric','Mean improvement','95\% CI'],
      [[label,f"{100*joint[k]['mean']:.2f}"+unit,
        f"[{100*joint[k]['ci95'][0]:.2f}, {100*joint[k]['ci95'][1]:.2f}]"+unit] for k,label,unit in [
        ('action_mse_improvement','Action MSE reduction',r'\%'),('next_slice_improvement','Sensory MSE reduction',r'\%'),
        ('semantic_accuracy_change','Semantic accuracy change',' points'),('binding_r1_change','Image--text retrieval change',' points')]])
bridge=read('formation_bridge.json')
table('table_bridge','One-foundation training-stage pilot. MSE is evaluated on 126 shifted transitions after the observation prefixes. All variants inherit sensory predictive pretraining.','tab:bridge','lrrrr',
      ['Training stage','Frozen MSE','Score MSE','Full MSE','Gain (\%)'],
      [[{'sequential':'Sequential','action_only':'Action-only','joint':'Joint','no_next':'Joint, no-next'}[name]]+[f"{r['mse'][k]:.4f}" for k in ['frozen','score','full']]+[f"{100*r['gain']:.2f}"] for name,r in bridge['summary'].items()])
extra=read('additional_protocol/summary.json')['summary']
table('table_additional','Separate output-calibration development protocol: one sensory foundation, four paired units per method. These results are not pooled with the main-stream comparison.','tab:additional','lrrrr',
      ['Correction','MSE','Gain (\%)','Bytes','Update (ms)'],
      [[{'state':'Hidden residual state','output_nlms':'Output NLMS','output_rls':'Output RLS','lora_head':'Low-rank readout'}[k],f"{r['mse']:.4f}",f"{100*r['gain']:.2f}",f"{r['bytes']:,}",f"{r['median_ms']:.2f}"] for k,r in extra.items()])
probes=read('formation_probes.json')['evidence']
probe_rows=[]
probe_labels={
    'final_full_nll':('Temporal','Ordered sequence','NLL',1),
    'final_rate_nll':('Temporal','Rate-grid control','NLL',1),
    'final_order_nll':('Temporal','Shuffled order','NLL',1),
    'learned_prediction_loss':('Future statistics','Learned encoder','Loss',1),
    'random_prediction_loss':('Future statistics','Random encoder','Loss',1),
    'ia_joint_top1':('Factor association','True pairing','Accuracy (\\%)',100),
    'ia_shuffled_joint_top1':('Factor association','Shuffled pairing','Accuracy (\\%)',100)}
for key,fields in [('M60',['final_full_nll','final_rate_nll','final_order_nll']),('M69',['learned_prediction_loss','random_prediction_loss']),('M71',['ia_joint_top1','ia_shuffled_joint_top1'])]:
    for f in fields:
        study,condition,metric,factor=probe_labels[f]
        probe_rows.append([study,condition,metric,f"{factor*probes[key][f]['mean']:.4f}"])
table('table_probes','Controlled event-model probes, three-seed means. Lower NLL/loss and higher accuracy are better. Temporal, future-statistic and factor-association studies correspond to archived M60, M69 and M71 records.','tab:probes','lllr',['Study','Condition','Metric','Mean'],probe_rows)
(OUT/'FIGURE_DATA_SOURCES.json').write_text(json.dumps(records,indent=2),encoding='utf-8')

# Control labels follow actual stored arm identifiers.
arm_labels={'frozen':'Frozen','safe_fast':'External state','full_tuning':'Prediction-model tuning (AdamW)',
            'scalar_rls':'Scalar RLS','replay':'Gain-memory replay','oracle':'Oracle inverse (uncapped correction)',
            'rls':'Scalar RLS','oracle_inverse':'Oracle inverse'}
table('table_control','Closed-loop attenuation: 75 trajectories per arm; adaptation carries across replicas within each model/task pair. Reward and success are evaluation-only. Controller definitions appear in Appendix~\\ref{app:control}.','tab:control','lrr',
      ['Method','Mean reward','Success (\%)'],
      [[arm_labels.get(k,k.replace('_',' ')),f"{r['mean_reward']:.2f}",f"{100*r['success_rate']:.2f}"] for k,r in control['arms'].items()])
curves=control['adaptation_curve']; state_key=next(k for k in curves if 'fast' in k)
xs=[r['decision'] for r in curves['frozen']]
delta=[a['mean_cumulative_reward']-b['mean_cumulative_reward'] for a,b in zip(curves[state_key],curves['frozen'])]
fig,axs=plt.subplots(1,2,figsize=(10,3.6))
axs[0].axvspan(.5,4.5,color='#e6ebf0');axs[0].plot(xs,delta,color=TEAL,marker='o',ms=4)
axs[0].axhline(0,color=GRAY,lw=.7);axs[0].set(xlabel='Completed decision',ylabel='Reward gain over frozen',xlim=(.5,max(xs)+.5))
axs[0].set_title('A  Improvement follows observation',loc='left',fontweight='bold')
# Derive task means directly from paired recorded trajectories.
raw_control=json.loads((DATA/'closed_loop_control.json').read_text())
tasks=sorted({r['task'] for r in raw_control['rows']})
task_gains=[]
for task in tasks:
    lookup={(r['model_seed'],r['replica'],r['arm']):r['cumulative_reward']
            for r in raw_control['rows'] if r['task']==task}
    pairs=[lookup[(seed,rep,'safe_fast')]-value for (seed,rep,arm),value in lookup.items() if arm=='frozen']
    task_gains.append(mean(pairs))
axs[1].bar([t.replace('-v3','') for t in tasks],task_gains,color=TEAL)
axs[1].set(ylabel='Mean paired reward gain');axs[1].set_title('B  Task-level reward gains',loc='left',fontweight='bold')
fig.tight_layout(w_pad=2);save(fig,'fig4_control')
