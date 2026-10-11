const API_BASE = "http://127.0.0.1:8000";

// --- Field definitions, mapped to the exact keys each trained model expects ---
const KIDNEY_FIELDS = [
  ["bp","Blood Pressure (mm/Hg)","number",80],["sg","Specific Gravity of Urine","number:0.001",1.02],
  ["al","Albumin in Urine (0-5)","number",0],["su","Sugar in Urine (0-5)","number",0],
  ["rbc","Red Blood Cells in Urine","select:normal,abnormal","normal"],
  ["pc","Pus Cells in Urine","select:normal,abnormal","normal"],
  ["pcc","Pus Cell Clumps","select:notpresent,present","notpresent"],
  ["ba","Bacteria in Urine","select:notpresent,present","notpresent"],
  ["bgr","Random Blood Glucose \u2013 kidney panel (mg/dl)","number",90],
  ["bu","Blood Urea (mg/dl)","number",15],["sc","Serum Creatinine (mg/dl)","number:0.1",0.9],
  ["sod","Sodium (mEq/L)","number",140],["pot","Potassium (mEq/L)","number:0.1",4.2],
  ["hemo","Hemoglobin (g)","number:0.1",14.5],["pcv","Packed Cell Volume (%)","number",42],
  ["wc","White Blood Cell Count","number",7000],["rc","Red Blood Cell Count (millions/cumm)","number:0.1",4.8],
  ["htn","Hypertension (history)","select:no,yes","no"],["dm","Diabetes Mellitus (history)","select:no,yes","no"],
  ["cad","Coronary Artery Disease","select:no,yes","no"],["appet","Appetite","select:good,poor","good"],
  ["pe","Pedal Edema","select:no,yes","no"],["ane","Anemia","select:no,yes","no"],
];
const LIVER_FIELDS = [
  ["Gender","Gender","select:Male,Female","Male"],
  ["Total_Bilirubin","Total Bilirubin","number:0.1",0.5],["Direct_Bilirubin","Direct Bilirubin","number:0.1",0.1],
  ["Alkaline_Phosphotase","Alkaline Phosphotase","number",90],
  ["Alamine_Aminotransferase","ALT (Alamine Aminotransferase)","number",22],
  ["Aspartate_Aminotransferase","AST (Aspartate Aminotransferase)","number",24],
  ["Total_Protiens","Total Proteins","number:0.1",7.0],["Albumin","Albumin \u2013 liver panel","number:0.1",4.3],
  ["Albumin_and_Globulin_Ratio","Albumin/Globulin Ratio","number:0.01",1.3],
];
const DIABETES_FIELDS = [
  ["Pregnancies","Pregnancies (0 if male/none)","number",0],
  ["Glucose","Glucose \u2013 diabetes panel (mg/dl)","number",90],
  ["BloodPressure","Blood Pressure \u2013 diabetes panel","number",75],
  ["SkinThickness","Skin Thickness (mm)","number",22],["Insulin","Insulin","number",85],
  ["BMI","BMI","number:0.1",23.0],["DiabetesPedigreeFunction","Diabetes Pedigree Function","number:0.01",0.3],
];
const KNOWN_CONDITIONS = [
  ["diabetes","Known diabetic"],["kidney_disease","Known kidney condition"],
  ["liver_disease","Known liver condition"],["cardiovascular_disease","Known heart condition"],
];

function renderFields(container, defs){
  container.innerHTML = defs.map(([key,label,type,def])=>{
    if(type.startsWith("select:")){
      const opts = type.split(":")[1].split(",");
      return `<div><label>${label}</label><select data-key="${key}">
        ${opts.map(o=>`<option value="${o}" ${o===def?"selected":""}>${o}</option>`).join("")}</select></div>`;
    }
    const step = type.includes(":") ? type.split(":")[1] : "1";
    return `<div><label>${label}</label><input data-key="${key}" type="number" step="${step}" value="${def ?? ""}"></div>`;
  }).join("");
  container.className = "grid";
}

// Loads a preset value object {key: value} into whichever fields exist in a container
function applyPreset(container, values){
  container.querySelectorAll("[data-key]").forEach(el=>{
    if(values[el.dataset.key] !== undefined) el.value = values[el.dataset.key];
  });
}

const PRESETS = {
  highRisk: {
    kidney: {bp:150, sg:1.01, al:3, su:2, rbc:"abnormal", pc:"abnormal", pcc:"present", ba:"notpresent",
      bgr:180, bu:65, sc:3.1, sod:135, pot:5.2, hemo:9.2, pcv:28, wc:9800, rc:3.8,
      htn:"yes", dm:"yes", cad:"no", appet:"poor", pe:"yes", ane:"yes"},
    liver: {Gender:"Male", Total_Bilirubin:2.1, Direct_Bilirubin:0.9, Alkaline_Phosphotase:250,
      Alamine_Aminotransferase:80, Aspartate_Aminotransferase:95, Total_Protiens:6.0, Albumin:2.8,
      Albumin_and_Globulin_Ratio:0.7},
    diabetes: {Pregnancies:2, Glucose:190, BloodPressure:95, SkinThickness:30, Insulin:160, BMI:34.5,
      DiabetesPedigreeFunction:0.8},
    age: 55,
  },
  healthy: {
    kidney: {bp:80, sg:1.02, al:0, su:0, rbc:"normal", pc:"normal", pcc:"notpresent", ba:"notpresent",
      bgr:90, bu:15, sc:0.9, sod:140, pot:4.2, hemo:14.5, pcv:42, wc:7000, rc:4.8,
      htn:"no", dm:"no", cad:"no", appet:"good", pe:"no", ane:"no"},
    liver: {Gender:"Female", Total_Bilirubin:0.4, Direct_Bilirubin:0.1, Alkaline_Phosphotase:80,
      Alamine_Aminotransferase:18, Aspartate_Aminotransferase:20, Total_Protiens:7.2, Albumin:4.6,
      Albumin_and_Globulin_Ratio:1.5},
    diabetes: {Pregnancies:1, Glucose:88, BloodPressure:75, SkinThickness:22, Insulin:85, BMI:22.0,
      DiabetesPedigreeFunction:0.3},
    age: 30,
  },
};

function collectFields(container){
  const out = {};
  container.querySelectorAll("[data-key]").forEach(el=>{
    const v = el.value;
    out[el.dataset.key] = el.tagName === "SELECT" ? v : (v === "" ? null : Number(v));
  });
  return out;
}

// --- localStorage patient records (frontend-only; backend has no DB yet) ---
function getPatients(){ return JSON.parse(localStorage.getItem("shp_patients") || "[]"); }
function savePatients(list){ localStorage.setItem("shp_patients", JSON.stringify(list)); }
function addPatient(record){
  const list = getPatients(); list.unshift(record); savePatients(list); return record;
}
function getPatient(id){ return getPatients().find(p=>p.id===id); }
function updatePatientResult(id, result){
  const list = getPatients();
  const p = list.find(p=>p.id===id);
  if(p){ p.lastResult = result; p.lastRunAt = new Date().toISOString(); savePatients(list); }
}

async function callPredictFull(patientObj, labObj, conditions, fileOrNull){
  const fd = new FormData();
  fd.append("patient", JSON.stringify(patientObj));
  fd.append("laboratory", JSON.stringify(labObj));
  fd.append("known_conditions", JSON.stringify(conditions));
  if(fileOrNull) fd.append("urine_image", fileOrNull);
  const res = await fetch(`${API_BASE}/predict/full`, { method:"POST", body: fd });
  if(!res.ok) throw new Error(await res.text());
  return res.json();
}
