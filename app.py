import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from ortools.sat.python import cp_model
from datetime import time
import random

# ============================================
# MAIN SCHEDULER CLASS
# ============================================
class StudyScheduler:
    def __init__(self, target_hours, min_break, class_buffer, study_sessions, fixed_modules, assignments, day_start=450, day_end=1065):
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        
        self.TARGET_WEEKLY_HOURS = target_hours
        self.MIN_BREAK = min_break
        self.CLASS_BUFFER = class_buffer
        self.DAY_START = day_start  
        self.DAY_END = day_end      
        
        self.days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        self.day_to_int = {day: i for i, day in enumerate(self.days)}
        
        self.module_colors = {
            'BWIN817': '#FF6B6B', 'BWIB817': '#4ECDC4', 'BWIB822': '#45B7D1',
            'BWIM815': '#96CEB4', 'BWIB818': '#FFE194', 'LECTURE': '#E0E0E0',
            'BREAK': '#FFFFFF'
        }
        
        self.fixed_modules = fixed_modules
        self.study_sessions = study_sessions
        self.assignments = assignments
        self.solution_schedule = None

        # Dynamically assign colors
        for module, _, _, _ in self.study_sessions:
            if module not in self.module_colors:
                self.module_colors[module] = f"#{random.randint(0, 0xFFFFFF):06x}"
                
        # Updated to unpack 4 variables for assignments
        for name, _, _, _ in self.assignments:
            if name not in self.module_colors:
                self.module_colors[name] = f"#{random.randint(0, 0xFFFFFF):06x}"
        
    def create_model(self):
        self.study_intervals = {}
        self.active_vars = {}
        self.start_vars = {}
        self.end_vars = {}
        
        # 1. Flexible Study Sessions
        for module, min_s, max_s, duration in self.study_sessions:
            for session_num in range(max_s):
                for day in self.days:
                    key = (module, session_num, day)
                    interval_id = f"study_{module}_{session_num}_{day}"
                    
                    start_var = self.model.NewIntVar(self.DAY_START, self.DAY_END - duration, f"start_{interval_id}")
                    end_var = self.model.NewIntVar(self.DAY_START + duration, self.DAY_END, f"end_{interval_id}")
                    active_var = self.model.NewBoolVar(f"active_{interval_id}")
                    
                    interval = self.model.NewOptionalIntervalVar(start_var, duration, end_var, active_var, interval_id)
                    
                    self.study_intervals[key] = interval
                    self.active_vars[key] = active_var
                    self.start_vars[key] = start_var
                    self.end_vars[key] = end_var

        # 2. Strict Assignments (Now with a Start Day!)
        for name, start_day, due_day, total_hours in self.assignments:
            duration = 60 
            for session_num in range(total_hours):
                for day in self.days:
                    # ONLY create variables for days BETWEEN start_day and due_day (inclusive)
                    if self.day_to_int[start_day] <= self.day_to_int[day] <= self.day_to_int[due_day]:
                        key = (name, session_num, day)
                        interval_id = f"assign_{name}_{session_num}_{day}"
                        
                        start_var = self.model.NewIntVar(self.DAY_START, self.DAY_END - duration, f"start_{interval_id}")
                        end_var = self.model.NewIntVar(self.DAY_START + duration, self.DAY_END, f"end_{interval_id}")
                        active_var = self.model.NewBoolVar(f"active_{interval_id}")
                        
                        interval = self.model.NewOptionalIntervalVar(start_var, duration, end_var, active_var, interval_id)
                        
                        self.study_intervals[key] = interval
                        self.active_vars[key] = active_var
                        self.start_vars[key] = start_var
                        self.end_vars[key] = end_var
    
    def add_hard_constraints(self):
        all_active_minutes = []
                
        # Regular Study Constraints
        for module, min_s, max_s, duration in self.study_sessions:
            module_active_sessions = []
            for session_num in range(max_s):
                session_days = [self.active_vars[(module, session_num, day)] for day in self.days]
                is_scheduled = self.model.NewBoolVar(f"is_scheduled_{module}_{session_num}")
                self.model.Add(sum(session_days) == is_scheduled)
                module_active_sessions.append(is_scheduled)
                all_active_minutes.append(is_scheduled * duration)
                
                if session_num > 0:
                    self.model.AddImplication(is_scheduled, module_active_sessions[session_num - 1])
            self.model.Add(sum(module_active_sessions) >= min_s)
            
        # Assignment Constraints (Enforcing the window)
        for name, start_day, due_day, total_hours in self.assignments:
            duration = 60
            for session_num in range(total_hours):
                session_days = []
                for day in self.days:
                    if self.day_to_int[start_day] <= self.day_to_int[day] <= self.day_to_int[due_day]:
                        session_days.append(self.active_vars[(name, session_num, day)])
                
                # Hard Constraint: MUST be scheduled exactly once within the valid window
                if session_days: # Safety check
                    self.model.AddExactlyOne(session_days)
                all_active_minutes.append(duration) 

        # Global Hours Target
        target_minutes = self.TARGET_WEEKLY_HOURS * 60
        self.model.Add(sum(all_active_minutes) >= target_minutes)
        
        # No Overlap per day
        for day in self.days:
            day_intervals = []
            for key, interval in self.study_intervals.items():
                if key[2] == day:
                    day_intervals.append(interval)
            for fixed_day, fixed_start, fixed_end, module_code in self.fixed_modules:
                if fixed_day == day:
                    fixed_int = self.model.NewIntervalVar(fixed_start, fixed_end - fixed_start, fixed_end, f"fixed_{module_code}_{day}_{fixed_start}")
                    day_intervals.append(fixed_int)
            if day_intervals:
                self.model.AddNoOverlap(day_intervals)
    
    def add_break_constraints(self):
        keys = list(self.active_vars.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                key1, key2 = keys[i], keys[j]
                if key1[2] == key2[2]:
                    act1, act2 = self.active_vars[key1], self.active_vars[key2]
                    start1, end1 = self.start_vars[key1], self.end_vars[key1]
                    start2, end2 = self.start_vars[key2], self.end_vars[key2]
                    
                    first_var = self.model.NewBoolVar(f"first_{i}_{j}")
                    self.model.Add(end1 + self.MIN_BREAK <= start2).OnlyEnforceIf([act1, act2, first_var])
                    self.model.Add(end2 + self.MIN_BREAK <= start1).OnlyEnforceIf([act1, act2, first_var.Not()])
        
        for key, active_var in self.active_vars.items():
            start_var, end_var, day = self.start_vars[key], self.end_vars[key], key[2]
            for fixed_day, fixed_start, fixed_end, module_code in self.fixed_modules:
                if fixed_day == day:
                    lec_id = f"{module_code}_{fixed_start}"
                    b_before = self.model.NewBoolVar(f"b_before_{key}_{lec_id}")
                    b_after = self.model.NewBoolVar(f"b_after_{key}_{lec_id}")
                    self.model.Add(end_var <= fixed_start - self.CLASS_BUFFER).OnlyEnforceIf(b_before)
                    self.model.Add(start_var >= fixed_end + self.CLASS_BUFFER).OnlyEnforceIf(b_after)
                    self.model.AddBoolOr([b_before, b_after]).OnlyEnforceIf(active_var)
    
    def add_objective(self):
        objective_terms = []
        for key, active_var in self.active_vars.items():
            start_var = self.start_vars[key]
            start_penalty = self.model.NewIntVar(0, self.DAY_END, f"penalty_{key}")
            self.model.Add(start_penalty == start_var - self.DAY_START).OnlyEnforceIf(active_var)
            self.model.Add(start_penalty == 0).OnlyEnforceIf(active_var.Not())
            objective_terms.append(start_penalty)
        self.model.Minimize(sum(objective_terms))
    
    def solve(self):
        self.solver.parameters.max_time_in_seconds = 10.0
        status = self.solver.Solve(self.model)
        if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
            self.solution_schedule = self._extract_solution()
            return self.solution_schedule, status
        return None, status
    
    def _extract_solution(self):
        schedule = []
        for key, active_var in self.active_vars.items():
            if self.solver.Value(active_var):
                module, session_num, day = key
                start, end = self.solver.Value(self.start_vars[key]), self.solver.Value(self.end_vars[key])
                schedule.append({'module': module, 'day': day, 'start': self._min_to_t(start), 'end': self._min_to_t(end), 'start_min': start, 'end_min': end, 'type': 'study'})
        for d, s, e, m in self.fixed_modules:
            schedule.append({'module': m, 'day': d, 'start': self._min_to_t(s), 'end': self._min_to_t(e), 'start_min': s, 'end_min': e, 'type': 'lecture'})
        
        schedule.sort(key=lambda x: (self.day_to_int[x['day']], x['start_min']))
        return schedule
    
    def _min_to_t(self, minutes):
        return f"{minutes // 60:02d}:{minutes % 60:02d}"

    def get_matplotlib_figure(self):
        if not self.solution_schedule: return None
        fig, ax = plt.subplots(figsize=(15, 8))
        time_slots = np.arange(self.DAY_START, self.DAY_END + 30, 30)
        time_labels = [self._min_to_t(t) for t in time_slots[::2]]
        
        for i, day in enumerate(self.days):
            for t in time_slots: ax.axhline(y=t, xmin=i/5, xmax=(i+1)/5, color='lightgray', linewidth=0.5, alpha=0.3)
            ax.axvline(x=i, color='black', linewidth=1)
            
        for s in self.solution_schedule:
            day_idx, start, end = self.day_to_int[s['day']], s['start_min'], s['end_min']
            color = self.module_colors.get('LECTURE' if s['type'] == 'lecture' else s['module'], '#999999')
            alpha = 0.5 if s['type'] == 'lecture' else 0.8
            ax.add_patch(plt.Rectangle((day_idx - 0.45, start), 0.9, end - start, facecolor=color, edgecolor='black', alpha=alpha, zorder=3))
            if (end - start) >= 45:
                label = s['module'].replace('LECTURE_', 'LEC ') if s['type'] == 'lecture' else s['module']
                ax.text(day_idx, start + (end - start)/2, label, ha='center', va='center', fontsize=8, fontweight='bold', color='black' if s['type'] == 'lecture' else 'white', zorder=4)
                
        ax.set_xlim(-0.5, 4.5); ax.set_ylim(self.DAY_END, self.DAY_START)
        ax.set_xticks(range(5)); ax.set_xticklabels(self.days, fontsize=11, fontweight='bold')
        ax.set_yticks(time_slots[::2]); ax.set_yticklabels(time_labels)
        ax.set_ylabel('Time', fontsize=11)
        ax.set_title(f'Weekly Study Schedule ({self.TARGET_WEEKLY_HOURS} Hours Target)', fontsize=14, fontweight='bold', pad=20)
        return fig

# ============================================
# HELPER FUNCTIONS FOR UI
# ============================================
def time_str_to_min(time_str):
    try:
        h, m = map(int, str(time_str).split(':'))
        return h * 60 + m
    except:
        return 0

# ============================================
# STREAMLIT APP
# ============================================
st.set_page_config(page_title="Study Scheduler", layout="wide", page_icon="📅")

st.title("📚 Intelligent Study Schedule Optimizer")
st.markdown("Set your constraints, define your modules, and let the AI build your perfect week.")

with st.sidebar:
    st.header("⚙️ Global Constraints")
    target_hours = st.number_input("Target Study Hours / Week", min_value=1, max_value=40, value=15, step=1)
    min_break = st.number_input("Minimum Break (mins)", min_value=0, max_value=120, value=45, step=15)
    class_buffer = st.number_input("Class Buffer (mins)", min_value=0, max_value=120, value=45, step=15)
    
    st.markdown("---")
    st.header("🕒 Day Boundaries")
    day_start_str = st.time_input("Day Starts At", value=time(7, 30))
    day_end_str = st.time_input("Day Ends At", value=time(17, 45))
    day_start = day_start_str.hour * 60 + day_start_str.minute
    day_end = day_end_str.hour * 60 + day_end_str.minute

# --- MAIN AREA DATA EDITORS ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎓 Fixed Lectures")
    default_lectures = pd.DataFrame([
        {'Day': 'Monday', 'Start Time': '11:00', 'End Time': '12:45', 'Module Code': 'LECTURE_BWIN817'},
        {'Day': 'Monday', 'Start Time': '13:00', 'End Time': '14:15', 'Module Code': 'LECTURE_BWIB817'},
        {'Day': 'Tuesday', 'Start Time': '11:00', 'End Time': '12:45', 'Module Code': 'LECTURE_BWIN817'},
        {'Day': 'Tuesday', 'Start Time': '13:00', 'End Time': '14:15', 'Module Code': 'LECTURE_BWIB822'},
        {'Day': 'Wednesday', 'Start Time': '09:30', 'End Time': '10:45', 'Module Code': 'LECTURE_BWIM815'},
        {'Day': 'Thursday', 'Start Time': '11:00', 'End Time': '12:45', 'Module Code': 'LECTURE_BWIB818'},
        {'Day': 'Friday', 'Start Time': '09:30', 'End Time': '10:45', 'Module Code': 'LECTURE_BWIM815'}
    ])
    edited_lectures = st.data_editor(default_lectures, num_rows="dynamic", width="stretch")

with col2:
    st.subheader("📖 Self-Study Requirements")
    default_study = pd.DataFrame([
        {'Module': 'BWIN817', 'Min Sessions': 2, 'Max Sessions': 4, 'Duration (mins)': 120},
        {'Module': 'BWIB817', 'Min Sessions': 2, 'Max Sessions': 5, 'Duration (mins)': 60},
        {'Module': 'BWIB822', 'Min Sessions': 2, 'Max Sessions': 5, 'Duration (mins)': 90},
        {'Module': 'BWIB818', 'Min Sessions': 2, 'Max Sessions': 5, 'Duration (mins)': 60}
    ])
    edited_study = st.data_editor(default_study, num_rows="dynamic", width="stretch")

st.markdown("---")
st.subheader("📝 Assignments & Deadlines")
st.markdown("The solver will break the total hours into 1-hour blocks and schedule them **between** your Start and Due days.")

default_assignments = pd.DataFrame([
    {'Assignment Name': 'SAS Project', 'Start Day': 'Wednesday', 'Due Day': 'Thursday', 'Total Hours Needed': 4}
])

edited_assignments = st.data_editor(
    default_assignments, 
    num_rows="dynamic", 
    width="stretch",
    column_config={
        "Start Day": st.column_config.SelectboxColumn(
            "Start Day",
            help="The earliest day you want to work on this",
            options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
            required=True
        ),
        "Due Day": st.column_config.SelectboxColumn(
            "Due Day",
            help="The deadline day (it must be finished on or before this day)",
            options=['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'],
            required=True
        )
    }
)

# --- EXECUTION ---
st.markdown("---")
if st.button("🚀 Generate Optimal Schedule", type="primary", width="stretch"):
    with st.spinner("AI is solving constraints..."):
        
        fixed_modules_list = []
        for _, row in edited_lectures.dropna().iterrows():
            fixed_modules_list.append((row['Day'], time_str_to_min(row['Start Time']), time_str_to_min(row['End Time']), row['Module Code']))
            
        study_sessions_list = []
        for _, row in edited_study.dropna().iterrows():
            study_sessions_list.append((row['Module'], int(row['Min Sessions']), int(row['Max Sessions']), int(row['Duration (mins)'])))
            
        assignments_list = []
        for _, row in edited_assignments.dropna().iterrows():
            # Pass the new Start Day to the solver
            assignments_list.append((row['Assignment Name'], row['Start Day'], row['Due Day'], int(row['Total Hours Needed'])))
            
        scheduler = StudyScheduler(
            target_hours=target_hours, 
            min_break=min_break, 
            class_buffer=class_buffer,
            study_sessions=study_sessions_list,
            fixed_modules=fixed_modules_list,
            assignments=assignments_list,
            day_start=day_start,
            day_end=day_end
        )
        
        scheduler.create_model()
        scheduler.add_hard_constraints()
        scheduler.add_break_constraints()
        scheduler.add_objective()
        
        schedule_data, status = scheduler.solve()
        
        if schedule_data:
            st.success(f"Successfully generated a valid schedule! (Status: {scheduler.solver.StatusName(status)})")
            fig = scheduler.get_matplotlib_figure()
            if fig:
                st.pyplot(fig)
            
            st.subheader("📋 Schedule Data")
            df_schedule = pd.DataFrame(schedule_data)[['day', 'start', 'end', 'module', 'type']]
            df_schedule.rename(columns={'day': 'Day', 'start': 'Start', 'end': 'End', 'module': 'Module', 'type': 'Type'}, inplace=True)
            st.dataframe(df_schedule, width="stretch")
            
            csv = df_schedule.to_csv(index=False).encode('utf-8')
            st.download_button("⬇️ Download as CSV", data=csv, file_name="optimized_schedule.csv", mime="text/csv")
            
        else:
            st.error(f"❌ INFEASIBLE: The AI could not find a valid schedule. Make sure your Start Day isn't after your Due Day!")