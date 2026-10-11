// Rule-based clinical interpretation. Standard adult reference ranges; decision support only.
// r = [low, high] or {M:[..],F:[..]}; c = critical [low, high]; t = severity cut-offs (fraction beyond limit); s = scale for zero-based ranges.
const REF = [
 {k:"bp",n:"Blood pressure",u:"mm/Hg",g:"Kidney",r:[60,90],c:[40,120],h:"Raised pressure both damages and results from kidney disease; needs BP control and medication review.",l:"Low pressure can reduce kidney perfusion; check hydration and drugs."},
 {k:"sg",n:"Urine specific gravity",u:"",g:"Kidney",r:[1.005,1.03],h:"Concentrated urine, usually dehydration; glucose or protein in urine also raises it.",l:"Dilute urine may mean reduced concentrating ability of the kidneys."},
 {k:"al",n:"Urine albumin",u:"grade 0-5",g:"Kidney",r:[0,0],s:8,h:"Albumin in urine is an early marker of kidney damage; confirm with urine albumin-creatinine ratio."},
 {k:"su",n:"Urine sugar",u:"grade 0-5",g:"Kidney",r:[0,0],s:8,h:"Glucose in urine suggests high blood sugar or a lowered kidney threshold; check blood glucose."},
 {k:"bgr",n:"Random blood glucose",u:"mg/dl",g:"Kidney",r:[70,140],c:[50,400],h:"Elevated glucose; confirm with fasting glucose or HbA1c to assess for diabetes.",l:"Low glucose; check medication effect and fasting state."},
 {k:"bu",n:"Blood urea",u:"mg/dl",g:"Kidney",r:[7,20],h:"Raised urea points to reduced kidney clearance, dehydration or high protein breakdown.",l:"Usually insignificant; can reflect low protein intake or liver disease."},
 {k:"sc",n:"Serum creatinine",u:"mg/dl",g:"Kidney",r:{M:[0.7,1.3],F:[0.6,1.1]},c:[0,5],h:"Raised creatinine indicates reduced filtration; repeat, calculate eGFR, review nephrotoxic drugs.",l:"Can reflect low muscle mass; rarely significant."},
 {k:"sod",n:"Sodium",u:"mEq/L",g:"Kidney",r:[135,145],c:[125,155],h:"High sodium suggests dehydration or fluid imbalance.",l:"Low sodium can cause confusion; review fluids, diuretics and kidney function."},
 {k:"pot",n:"Potassium",u:"mEq/L",g:"Kidney",r:[3.5,5],c:[3,6],h:"High potassium risks dangerous heart rhythms; obtain an ECG and review urgently if markedly raised.",l:"Low potassium causes weakness and arrhythmia risk; check diuretics and losses."},
 {k:"hemo",n:"Hemoglobin",u:"g/dL",g:"Kidney",r:{M:[13.5,17.5],F:[12,15.5]},c:[7,20],h:"Raised hemoglobin can reflect dehydration or polycythemia.",l:"Anemia is common in chronic kidney disease; check iron studies, B12 and folate."},
 {k:"pcv",n:"Packed cell volume",u:"%",g:"Kidney",r:{M:[41,53],F:[36,46]},h:"High PCV suggests hemoconcentration.",l:"Low PCV supports anemia; read with hemoglobin."},
 {k:"wc",n:"White cell count",u:"/cumm",g:"Kidney",r:[4000,11000],h:"Raised WBC suggests infection or inflammation.",l:"Low WBC may indicate marrow suppression or viral illness."},
 {k:"rc",n:"Red cell count",u:"mill/cumm",g:"Kidney",r:{M:[4.7,6.1],F:[4.2,5.4]},h:"High count may reflect dehydration.",l:"Low count is consistent with anemia."},
 {k:"Total_Bilirubin",n:"Total bilirubin",u:"mg/dL",g:"Liver",r:[0.1,1.2],t:[0.25,1],h:"Suggests liver dysfunction, bile duct obstruction or hemolysis; jaundice is likely above about 2.5."},
 {k:"Direct_Bilirubin",n:"Direct bilirubin",u:"mg/dL",g:"Liver",r:[0,0.3],t:[0.5,2],h:"Conjugated hyperbilirubinemia points to cholestasis or bile duct problems."},
 {k:"Alkaline_Phosphotase",n:"Alkaline phosphatase",u:"IU/L",g:"Liver",r:[44,147],t:[0.5,1.5],h:"Suggests bile duct obstruction or liver/bone disease.",l:"Rarely significant."},
 {k:"Alamine_Aminotransferase",n:"ALT",u:"U/L",g:"Liver",r:[7,56],t:[0.5,1.5],h:"Indicates liver cell injury (fatty liver, hepatitis, drugs); mild rises are common."},
 {k:"Aspartate_Aminotransferase",n:"AST",u:"U/L",g:"Liver",r:[10,40],t:[0.5,1.5],h:"Indicates liver or muscle injury; AST above ALT can suggest alcohol-related disease."},
 {k:"Total_Protiens",n:"Total protein",u:"g/dL",g:"Liver",r:[6,8.3],h:"May reflect dehydration or chronic inflammation.",l:"Suggests poor synthesis (liver), loss (kidney) or malnutrition."},
 {k:"Albumin",n:"Serum albumin",u:"g/dL",g:"Liver",r:[3.5,5],l:"Reflects reduced liver synthesis, protein loss in urine or poor nutrition."},
 {k:"Albumin_and_Globulin_Ratio",n:"Albumin/globulin ratio",u:"",g:"Liver",r:[1.1,2.5],l:"A low ratio can indicate chronic liver disease or inflammation."},
 {k:"Glucose",n:"Glucose (diabetes panel)",u:"mg/dl",g:"Metabolic",r:[70,140],c:[50,400],h:"Elevated; correlate with fasting glucose/HbA1c and symptoms.",l:"Low glucose; check medication and fasting state."},
 {k:"BMI",n:"BMI",u:"kg/m2",g:"Metabolic",r:[18.5,24.9],h:"Excess weight raises diabetes, kidney and cardiovascular risk; lifestyle review advised.",l:"Underweight; assess nutrition."},
];
const LEVELS = [
  {n:"Within range",a:"No action needed."},
  {n:"Minor",a:"Small deviation. Monitor; often not significant in isolation."},
  {n:"Moderate",a:"Repeat the test and correlate clinically."},
  {n:"Marked",a:"Bring to the physician's attention."},
  {n:"Critical",a:"Urgent clinical review advised."},
];
const FLAGS = [["rbc","abnormal","Abnormal RBC in urine"],["pc","abnormal","Abnormal pus cells"],["pcc","present","Pus cell clumps"],
 ["ba","present","Bacteria in urine"],["htn","yes","Hypertension"],["dm","yes","Diabetes history"],["cad","yes","Coronary artery disease"],
 ["appet","poor","Poor appetite"],["pe","yes","Pedal edema"],["ane","yes","Anemia"]];

function evaluateLabs(vals){
  const sex = vals.Gender === "Female" ? "F" : "M";
  return REF.filter(p => vals[p.k] != null && !isNaN(vals[p.k])).map(p => {
    const [lo, hi] = Array.isArray(p.r) ? p.r : p.r[sex];
    const v = Number(vals[p.k]);
    let level = 0, dir = "";
    if (v > hi || v < lo) {
      dir = v > hi ? "h" : "l";
      const pct = v > hi ? (v - hi) / (p.s || hi) : (lo - v) / lo;
      const t = p.t || [0.15, 0.4];
      level = pct < t[0] ? 1 : pct < t[1] ? 2 : 3;
      if (p.c && (v < p.c[0] || v > p.c[1])) level = 4;
    }
    const note = level ? (p[dir] || "Outside the reference range; interpret clinically.") : "";
    return {p, v, lo, hi, level, dir, note};
  });
}
function rangeText(r){ return r.hi === 0 && r.lo === 0 ? "0 (none)" : `${r.lo} - ${r.hi}`; }
function activeFlags(vals){ return FLAGS.filter(([k, bad]) => vals[k] === bad).map(f => f[2]); }
