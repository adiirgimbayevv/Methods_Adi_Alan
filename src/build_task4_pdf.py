from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate, Paragraph, Spacer, FrameBreak, Table, TableStyle, Image, Flowable, PageBreak, NextPageTemplate
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.utils import ImageReader

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output' / 'pdf' / 'Task4_IEEE_Paper.pdf'
OUT.parent.mkdir(parents=True, exist_ok=True)
PAGE_W, PAGE_H = letter
MARGIN = 0.58 * inch
GAP = 0.24 * inch
COL_W = (PAGE_W - 2*MARGIN - GAP) / 2
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='PaperTitle', fontName='Times-Bold', fontSize=18, leading=20, alignment=TA_CENTER, spaceAfter=7))
styles.add(ParagraphStyle(name='Authors', fontName='Times-Roman', fontSize=9, leading=11, alignment=TA_CENTER, spaceAfter=6))
styles.add(ParagraphStyle(name='Abstract', fontName='Times-Roman', fontSize=8.3, leading=9.5, alignment=TA_JUSTIFY, spaceAfter=3))
styles.add(ParagraphStyle(name='Keywords', fontName='Times-Roman', fontSize=8, leading=9, alignment=TA_JUSTIFY))
styles.add(ParagraphStyle(name='Body', fontName='Times-Roman', fontSize=10.0, leading=12.0, alignment=TA_JUSTIFY, spaceAfter=3))
styles.add(ParagraphStyle(name='Heading1x', fontName='Times-Bold', fontSize=9.6, leading=11, alignment=TA_CENTER, spaceBefore=5, spaceAfter=3, keepWithNext=True))
styles.add(ParagraphStyle(name='Heading2x', fontName='Times-BoldItalic', fontSize=8.8, leading=10, alignment=TA_LEFT, spaceBefore=3, spaceAfter=2, keepWithNext=True))
styles.add(ParagraphStyle(name='Caption', fontName='Times-Roman', fontSize=7.5, leading=8.4, alignment=TA_CENTER, spaceBefore=2, spaceAfter=2))
styles.add(ParagraphStyle(name='Ref', fontName='Times-Roman', fontSize=7.2, leading=8, alignment=TA_LEFT, leftIndent=11, firstLineIndent=-11, spaceAfter=2))

class Pipeline(Flowable):
    def __init__(self, width=COL_W, height=54):
        Flowable.__init__(self); self.width=width; self.height=height
    def draw(self):
        c=self.canv; labels=['Filtered test rows','Submission\nscores','Student-lab\ntrajectories','Arm summaries']
        gap=8; bw=(self.width-3*gap)/4; y=9; h=37
        for i,label in enumerate(labels):
            x=i*(bw+gap)
            c.setStrokeColor(colors.HexColor('#53677A')); c.setFillColor(colors.HexColor('#EEF3F7'))
            c.roundRect(x,y,bw,h,3,stroke=1,fill=1)
            c.setFillColor(colors.black); c.setFont('Times-Roman',6.8)
            for j,line in enumerate(label.split('\n')): c.drawCentredString(x+bw/2,y+h/2+3-j*9,line)
            if i<3:
                c.setStrokeColor(colors.HexColor('#53677A')); c.line(x+bw+1,y+h/2,x+bw+gap-2,y+h/2)
                c.line(x+bw+gap-5,y+h/2+2,x+bw+gap-2,y+h/2)
                c.line(x+bw+gap-5,y+h/2-2,x+bw+gap-2,y+h/2)

class GainChart(Flowable):
    def __init__(self,width,height=126): Flowable.__init__(self); self.width=width; self.height=height
    def draw(self):
        c=self.canv; left=34; right=self.width-8; bottom=30; top=self.height-18
        c.setFont('Times-Bold',7.4); c.drawCentredString(self.width/2,self.height-9,'Mean score gain after an initial failure (pp)')
        ymin,ymax=55.0,75.0
        def yy(v): return bottom+(v-ymin)/(ymax-ymin)*(top-bottom)
        for v in [55,60,65,70,75]:
            y=yy(v); c.setStrokeColor(colors.HexColor('#D9E0E6')); c.line(left,y,right,y); c.setFillColor(colors.black); c.setFont('Times-Roman',6.5); c.drawRightString(left-4,y-2,str(v))
        vals=[64.552,63.187,68.507]; los=[58.478,56.159,61.546]; his=[70.538,70.366,74.830]
        labels=['No feedback','Test-case hint','Natural-language']
        xs=[left+24,left+(right-left)/2,right-24]
        for x,v,lo,hi,label in zip(xs,vals,los,his,labels):
            c.setStrokeColor(colors.HexColor('#D88B00')); c.setLineWidth(1.1); c.line(x,yy(lo),x,yy(hi)); c.line(x-3,yy(lo),x+3,yy(lo)); c.line(x-3,yy(hi),x+3,yy(hi)); c.setFillColor(colors.HexColor('#222222')); c.circle(x,yy(v),2.5,stroke=0,fill=1); c.setFillColor(colors.black); c.setFont('Times-Roman',6.1); c.drawCentredString(x,14,label)
        c.setFillColor(colors.HexColor('#444444')); c.setFont('Times-Italic',6.2); c.drawString(0,self.height-17,'Bars: student-cluster bootstrap 95% intervals')

class PaperDoc(BaseDocTemplate):
    def __init__(self, filename):
        BaseDocTemplate.__init__(self, filename, pagesize=letter, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=0.48*inch, bottomMargin=0.48*inch)
        top=Frame(MARGIN, PAGE_H-3.05*inch, PAGE_W-2*MARGIN, 2.5*inch, id='title', leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        ybottom=0.48*inch; ytop=PAGE_H-3.25*inch
        left=Frame(MARGIN, ybottom, COL_W, ytop-ybottom, id='left', leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        right=Frame(MARGIN+COL_W+GAP, ybottom, COL_W, ytop-ybottom, id='right', leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        body_left=Frame(MARGIN,ybottom,COL_W,PAGE_H-0.48*inch-ybottom,id='bodyleft',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
        body_right=Frame(MARGIN+COL_W+GAP,ybottom,COL_W,PAGE_H-0.48*inch-ybottom,id='bodyright',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
        self.addPageTemplates([PageTemplate(id='first',frames=[top,left,right],onPage=self.decorate), PageTemplate(id='body',frames=[body_left,body_right],onPage=self.decorate)])
    def decorate(self, canvas, doc):
        canvas.saveState(); canvas.setFont('Times-Roman',7); canvas.setFillColor(colors.HexColor('#555555'))
        canvas.drawString(MARGIN,0.28*inch,'Task 4 | ProgFeed classroom analysis')
        canvas.drawRightString(PAGE_W-MARGIN,0.28*inch,str(doc.page))
        canvas.restoreState()

def P(text, style='Body'): return Paragraph(text,styles[style])
def section(text): return P(text,'Heading1x')
def sub(text): return P(text,'Heading2x')

story=[]
story += [P('AI-Generated Feedback and Short-Term Programming Outcomes A Descriptive Analysis of ProgFeed','PaperTitle'), P('Adi Irgimbayev and Alan Shmyrin | Research project, Task 4','Authors')]
abstract=('AI-generated feedback is increasingly used to help novice programmers, but evidence about the relative value of different feedback formats in authentic classrooms remains limited. We analyze ProgFeed, a de-identified dataset from an introductory Python course containing 17,385 graded-test rows, 6,693 submissions, and 215 consenting students. The analysis aggregates test rows into submissions and student-lab trajectories, then compares assigned natural-language hints, test-case hints, and no-feedback control in the four labs with randomized assignment. Among student-lab units whose first submission failed, mean score gain from first to final observed submission was 64.55 percentage points under no feedback, 63.19 under test-case hints, and 68.51 under natural-language hints. Student-cluster bootstrap intervals overlap, and retry and eventual-full-credit rates are high in all three arms. These are descriptive, short-term outcomes: feedback is only delivered after errors, final submissions may be incomplete, and this reanalysis does not establish causal effects or measure retention, transfer, or critical thinking. We provide a reproducible aggregation and uncertainty procedure and identify measurement choices needed for stronger follow-up studies.')
story += [P('<b>Abstract—</b>'+abstract,'Abstract'), P('<b>Index Terms—</b> AI in education, programming education, formative feedback, learning analytics, autograding.','Keywords'), FrameBreak(), NextPageTemplate('body')]
story += [section('I. INTRODUCTION'), P('Generative AI can produce code, explanations, and debugging advice, changing how novices seek help and complete programming exercises. Early evaluations of Codex and Copilot showed that models could solve many introductory tasks, raising instructional opportunities and assessment concerns [1]–[3]. Controlled classroom studies since then report mixed learning results: AI access can support code generation and comprehension under some conditions, while reliance on a chatbot does not consistently improve post-task performance [4]–[6]. Recent work has begun to compare feedback modalities directly, but authentic classroom evidence remains scarce [7].'), P('This gap matters because feedback is not a single treatment. Its effects depend on informational content, timing, and whether it helps learners close a task-level knowledge gap [8]–[10]. Programming systems can return failing tests or natural-language hints, but these may direct attention to different parts of the problem. Studies of code explanation and integrated notebook support further motivate designs that help students interpret errors rather than merely receive a solution [11], [12].'), P('We ask: among initially failing student-lab units in randomized intervention labs, how do assigned natural-language hints, test-case hints, and no feedback relate to retries, observed score gain, and eventual full credit? This paper contributes: (1) a reproducible student-lab aggregation of an authentic public dataset; (2) arm-level retry, full-credit, and score-gain estimates with student-cluster bootstrap intervals; and (3) an explicit distinction between assignment, delivered feedback, and short-term score change.'), section('II. METHODOLOGY'), sub('A. Dataset and design'), P('ProgFeed records execution-based scores, assigned feedback condition, feedback text, timestamps, and de-identified student identifiers from an introductory Python course [7]. Its public consolidated file contains 17,385 test/function rows from 215 consenting students across 6,693 submissions. We restrict comparisons to labs 05, 06, 07, and 09, the four labs with randomized assignment to no feedback, a generated failing-test hint (tc), or a natural-language hint (nl). Assigned arm is retained even when feedback was not delivered; the system generated hints only after an unsuccessful attempt.'), sub('B. Aggregation and outcomes'), P('The analysis unit is one student in one lab. For submission t, earned and available points are summed across graded rows. Equation (1) defines the score as a percentage of available test points; for an initially failing unit, Eq. (2) defines observed gain as the final score minus the first score.'), P('<b>(1)</b> &nbsp; Submission score = 100 &times; total earned points / total available points.'), P('<b>(2)</b> &nbsp; Score gain = final observed score &minus; first score, when first score is below 100%.'), P('Here i indexes a student-lab unit, t submission order, j a graded test/function, s earned points, m available points, and T<sub>i</sub> the last observed submission. A retry means at least one later submission; eventual full credit means any observed submission reached at least 99.999%. These definitions describe observed records, not mastery or delayed retention.'), P('Fig. 1 shows the processing path. We exclude nonconsenting rows and nonrandomized labs before grouping by student, lab, and timestamp, then order submissions and summarize by assigned arm.'), Pipeline(), P('Fig. 1. Analysis pipeline from test rows to student-lab outcomes.','Caption'), sub('C. Estimation'), P('We report arm-specific means among initially failing units, retry rates, and eventual-full-credit rates. A percentile 95% interval for mean G is calculated from 5,000 bootstrap draws (seed 42), resampling students and retaining all observed labs for each selected student. This preserves within-student clustering and matches the student-lab weighted point estimate. We do not report significance tests or causal contrasts: this reanalysis conditions on an initial failure and uses final observed submissions; assignment also differs from actual hint exposure.'), section('III. RESULTS AND DISCUSSION'), P('Table I summarizes 280 student-lab units with a failed first attempt (out of 803 units across the assigned arms). The natural-language arm has the largest observed mean gain (68.51 points), compared with 64.55 for no feedback and 63.19 for test-case hints. The student-cluster intervals overlap substantially. These descriptive results are compatible with prior evidence that feedback format and validity may matter, but this restricted reaggregation is not a causal replication of the source paper [7].')]

rows=[['Assigned arm','n','Retry / full credit','Mean gain (95% CI), pp'],['No feedback','93','97.8% / 91.4%','64.55 (58.48–70.54)'],['Test-case hint','91','95.6% / 90.1%','63.19 (56.16–70.37)'],['Natural-language','96','96.9% / 92.7%','68.51 (61.55–74.83)']]
t=Table(rows,colWidths=[1.05*inch,.35*inch,.82*inch,1.32*inch],repeatRows=1,hAlign='CENTER')
t.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Times-Roman'),('FONTNAME',(0,0),(-1,0),'Times-Bold'),('FONTSIZE',(0,0),(-1,-1),7.0),('LEADING',(0,0),(-1,-1),7.2),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E9EEF3')),('TEXTCOLOR',(0,0),(-1,-1),colors.black),('LINEBELOW',(0,0),(-1,0),0.7,colors.black),('LINEBELOW',(0,-1),(-1,-1),0.7,colors.black),('ALIGN',(1,1),(-1,-1),'CENTER'),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
story += [P('Table I reports observed outcomes for the initially failing subset.','Caption'),t, Spacer(1,4), P('Fig. 2 visualizes the same score-gain estimates and student-cluster intervals. The point differences are modest relative to the intervals; overlapping intervals alone are not a test of equivalence or a treatment effect. Nearly all initially failing units made another submission, and about nine in ten reached full credit at least once. This binary measure has limited room to distinguish arms and does not show whether students can solve a new task independently.'), GainChart(width=COL_W-10,height=126), P('Fig. 2. Mean score gain after an initial failure; intervals resample students.','Caption'), P('The pattern is consistent with prior evidence that feedback information content can change outcomes, while effects vary by intervention and context [8], [10]. Natural-language feedback may help localize an error, whereas failing tests make violated behavior explicit; neither format guarantees that a student can explain or reproduce a solution. Prior programming studies likewise distinguish assisted performance from later learning [4]–[6]. Future experiments should measure unaided post-tests, delayed retention, task transfer, and feedback validity, and should analyze assigned arm and actual hint exposure separately.'), section('IV. LIMITATIONS AND CONCLUSION'), P('This is a reanalysis of one course and four randomized labs, not a new intervention. The public data contain no validated critical-thinking scale, delayed retention test, or transfer task. Conditioning on initial failure can change comparability across arms; hints are conditional on errors; repeated labs within students require clustered uncertainty; and the last observed submission may be incomplete. Bootstrap intervals do not remove these design limitations or warrant causal language.'), P('Within this measurement envelope, all arms show high retry and eventual-full-credit rates, while the natural-language arm has a somewhat higher mean observed gain. These estimates suggest hypotheses for a prospective study, not evidence that a feedback type improves durable learning. Future work should pre-register outcomes, preserve assignment in analysis, record hint validity and exposure, and include unaided assessment. The source data are public; the repository provides a pinned downloader and scripts without committing student-level rows.'), PageBreak(), section('REFERENCES')]
refs=[
'[1] J. Finnie-Ansley, P. Denny, B. A. Becker, A. Luxton-Reilly, and J. Prather, “The robots are coming: Exploring the implications of OpenAI Codex on introductory programming,” in Proc. ACE, 2022, pp. 10–19, doi: 10.1145/3511861.3511863.',
'[2] M. Wermelinger, “Using GitHub Copilot to solve simple programming problems,” in Proc. SIGCSE, 2023, pp. 172–178, doi: 10.1145/3545945.3569830.',
'[3] B. A. Becker et al., “Programming is hard—or at least it used to be: Educational opportunities and challenges of AI code generation,” in Proc. SIGCSE, 2023, pp. 500–506, doi: 10.1145/3545945.3569759.',
'[4] M. Kazemitabaar et al., “Studying the effect of AI code generators on supporting novice learners in introductory programming,” in Proc. CHI, 2023, Art. 455, doi: 10.1145/3544548.3580919.',
'[5] Y. Xue et al., “Does ChatGPT help with introductory programming? An experiment of students using ChatGPT in CS1,” in Proc. ICSE-SEET, 2024, pp. 331–341, doi: 10.1145/3639474.3640076.',
'[6] D. Sun et al., “Would ChatGPT-facilitated programming mode impact college students’ programming behaviors, performances, and perceptions?,” Int. J. Educ. Technol. High. Educ., vol. 21, Art. 14, 2024, doi: 10.1186/s41239-024-00446-5.',
'[7] H. Heickal and A. Lan, “A classroom study of LLM-generated feedback intervention in introductory programming,” Proc. Mach. Learn. Res., vol. 339, pp. 12–21, 2026.',
'[8] J. Hattie and H. Timperley, “The power of feedback,” Rev. Educ. Res., vol. 77, no. 1, pp. 81–112, 2007, doi: 10.3102/003465430298487.',
'[9] V. J. Shute, “Focus on formative feedback,” Rev. Educ. Res., vol. 78, no. 1, pp. 153–189, 2008, doi: 10.3102/0034654307313795.',
'[10] B. Wisniewski, K. Zierer, and J. Hattie, “The power of feedback revisited: A meta-analysis of educational feedback research,” Front. Psychol., vol. 10, Art. 3087, 2020, doi: 10.3389/fpsyg.2019.03087.',
'[11] P. Denny et al., “Explaining code with a purpose: An integrated approach for developing code comprehension and prompting skills,” in Proc. ITiCSE, 2024, pp. 283–289, doi: 10.1145/3649217.3653587.',
'[12] K. Kuramitsu, Y. Obara, M. Sato, and M. Obara, “KOGI: A seamless integration of ChatGPT into Jupyter environments for programming education,” in Proc. SPLASH-E, 2023, pp. 50–59, doi: 10.1145/3622780.3623648.'
]
for r in refs: story.append(P(r,'Ref'))

param_rows=[['Analysis parameter','Specification'],['Comparison subset','Labs 05, 06, 07, 09; randomized assignment'],['Analysis unit','One student × one lab'],['Primary outcome','Final minus first score after a failed first attempt'],['Uncertainty','5,000 student-cluster bootstrap draws; seed 42'],['Exposure coding','Assigned arm retained; delivered hint recorded separately']]
param_table=Table(param_rows,colWidths=[1.05*inch,2.47*inch],repeatRows=1,hAlign='CENTER')
param_table.setStyle(TableStyle([('FONTNAME',(0,0),(-1,-1),'Times-Roman'),('FONTNAME',(0,0),(-1,0),'Times-Bold'),('FONTSIZE',(0,0),(-1,-1),7.0),('LEADING',(0,0),(-1,-1),8.0),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E9EEF3')),('LINEBELOW',(0,0),(-1,0),0.7,colors.black),('LINEBELOW',(0,-1),(-1,-1),0.7,colors.black),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
idx=next(i for i,x in enumerate(story) if isinstance(x,Paragraph) and x.getPlainText().startswith('III. RESULTS'))
story[idx-1:idx-1]=[P('For assigned arm a, the primary point estimate is the mean observed gain across initially failing student-lab units: G-bar(a) = sum of G for arm a divided by n(a). Retry and full-credit rates use the same initially failing denominator. Table II records the analysis settings needed to reproduce the comparison. No missing outcomes are imputed.','Body'),P('Table II. Reproducible analysis parameters.','Caption'),param_table]
idx=next(i for i,x in enumerate(story) if isinstance(x,Paragraph) and x.getPlainText().startswith('Within this measurement envelope'))
story.insert(idx,P('Several validity threats remain. Conditioning on an initially failed submission may select different students or tasks across arms if assignment affects early attempts. Student-cluster resampling addresses dependence among repeated labs but not this selection or unmeasured differences in lab difficulty. Score normalization uses available points as a denominator, yet it cannot make assignments equally difficult. The last observed attempt is a censored endpoint: students may stop before full credit, and the file does not establish whether they later learned the same concepts independently. Finally, hint quality and correctness are not measured uniformly for every assigned unit. The reported estimates therefore characterize observed course trajectories under this intervention and should not be generalized to other courses, models, or feedback policies without replication.','Body'))
idx=next(i for i,x in enumerate(story) if isinstance(x,Paragraph) and x.getPlainText().startswith('IV. LIMITATIONS'))
story.insert(idx,P('The result should be interpreted as progress during an assisted assignment, not as a learning gain in the experimental sense. The score reflects autograder points available on each lab, while the number and difficulty of tests can vary by lab. Aggregating to a percentage permits a common scale but does not equalize task difficulty. Student-cluster intervals address repeated student records; they do not adjust for lab-specific difficulty or selection into the initially failing subset.','Body'))
story.insert(idx+1,P('The intervention arms are also not equivalent to actual exposure: a student assigned to a hint condition may receive no hint until an error occurs, and the public records do not support a clean comparison of hint quality for all attempts. Future work should define exposure at the error event, validate hint content, and pair assignment-arm estimates with unaided transfer and retention measures.','Body'))

doc=PaperDoc(str(OUT)); doc.build(story)
print(OUT)
