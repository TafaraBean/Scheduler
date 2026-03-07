# ============================================
# STUDY SCHEDULE OPTIMIZER FOR BMI POSTGRADUATE STUDENTS
# ============================================
# Run this entire script in Google Colab or locally
# It will generate an optimized study schedule with visualizations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
from datetime import datetime, time
from ortools.sat.python import cp_model

# ============================================
# MAIN SCHEDULER CLASS
# ============================================

class StudyScheduler:
    def __init__(self):
        self.model = cp_model.CpModel()
        self.solver = cp_model.CpSolver()
        
        # 🎯 Set your total weekly target hours here!
        self.TARGET_WEEKLY_HOURS = 15
        
        # Time settings (in minutes from midnight)
        self.DAY_START = 7 * 60 + 30  # 07:30
        self.DAY_END = 17 * 60 + 45   # 17:45
        
        # Break & Buffer settings
        self.MIN_BREAK = 45           # 45-minute global minimum break between study sessions
        self.CLASS_BUFFER = 45        # 45-minute buffer before and after fixed lectures
        
        # Days mapping
        self.days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
        self.day_to_int = {day: i for i, day in enumerate(self.days)}
        
        # Color mapping for modules (for visualization)
        self.module_colors = {
            'BWIN817': '#FF6B6B',  # Red
            'BWIB817': '#4ECDC4',  # Teal
            'BWIB822': '#45B7D1',  # Blue
            'BWIM815': '#96CEB4',  # Green (Lectures only)
            'BWIB818': '#FFE194',  # Yellow
            'LECTURE': '#E0E0E0',  # Grey for fixed lectures
            'ABSA': '#D4A5A5',     # Pink for ABSA events
            'BREAK': '#FFFFFF'     # White for breaks
        }
        
        # Fixed MSc modules (from your timetable)
        self.fixed_modules = self._get_fixed_modules()
        
        # Study sessions to schedule
        self.study_sessions = self._define_study_sessions()
        
        # ABSA event (when it occurs)
        self.absa_dates = []  # List of specific Friday dates when ABSA happens
        
        # Store solution for visualization
        self.solution_schedule = None
        
    def _get_fixed_modules(self):
        """Fixed MSc modules from timetable"""
        return [
            ('Monday', 11*60, 12*60+45, 'LECTURE_BWIN817'),     
            ('Monday', 13*60, 14*60+15, 'LECTURE_BWIB817'),     
            ('Tuesday', 11*60, 12*60+45, 'LECTURE_BWIN817'),    
            ('Tuesday', 13*60, 14*60+15, 'LECTURE_BWIB822'),    
            ('Wednesday', 9*60+30, 10*60+45, 'LECTURE_BWIM815'),
            ('Thursday', 11*60, 12*60+45, 'LECTURE_BWIB818'),   
            ('Friday', 9*60+30, 10*60+45, 'LECTURE_BWIM815'),   
        ]
    
    def _define_study_sessions(self):
        """
        NEW FORMAT: (module, min_sessions, max_sessions, duration_in_minutes)
        Adjust the 4th parameter to change the length of the study block!
        """
        return [
            ('BWIN817', 2, 4, 120),  # 2-hour sessions
            ('BWIB817', 2, 5, 60),   # 1-hour sessions
            ('BWIB822', 2, 5, 90),   # 1.5-hour sessions
            ('BWIB818', 2, 5, 60),   # 1-hour sessions
        ]
    
    def create_model(self):
        """Create the constraint programming model"""
        self.study_intervals = {}
        self.active_vars = {}
        self.start_vars = {}
        self.end_vars = {}
        
        # NEW: Unpack the 4th parameter (duration)
        for module, min_s, max_s, duration in self.study_sessions:
            for session_num in range(max_s):
                for day in self.days:
                    key = (module, session_num, day)
                    interval_id = f"{module}_{session_num}_{day}"
                    
                    # Use the custom duration for start and end boundaries
                    start_var = self.model.NewIntVar(self.DAY_START, self.DAY_END - duration, f"start_{interval_id}")
                    end_var = self.model.NewIntVar(self.DAY_START + duration, self.DAY_END, f"end_{interval_id}")
                    active_var = self.model.NewBoolVar(f"active_{interval_id}")
                    
                    # Create the interval with the custom duration
                    interval = self.model.NewOptionalIntervalVar(
                        start_var, duration, end_var, active_var, interval_id
                    )
                    
                    self.study_intervals[key] = interval
                    self.active_vars[key] = active_var
                    self.start_vars[key] = start_var
                    self.end_vars[key] = end_var
    
    def add_hard_constraints(self):
        """Add all hard constraints"""
        
        all_active_minutes = []
        
        # CONSTRAINT 1: Handle variable number of study sessions per module
        for module, min_s, max_s, duration in self.study_sessions:
            module_active_sessions = []
            
            for session_num in range(max_s):
                session_days = [self.active_vars[(module, session_num, day)] for day in self.days]
                
                is_scheduled = self.model.NewBoolVar(f"is_scheduled_{module}_{session_num}")
                self.model.Add(sum(session_days) == is_scheduled)
                
                module_active_sessions.append(is_scheduled)
                
                # Multiply boolean by duration to calculate total minutes
                all_active_minutes.append(is_scheduled * duration)
                
                # Symmetry breaking
                if session_num > 0:
                    self.model.AddImplication(is_scheduled, module_active_sessions[session_num - 1])
                    
            # Enforce the absolute minimum required sessions for this specific module
            self.model.Add(sum(module_active_sessions) >= min_s)
            
        # CONSTRAINT 2: Enforce the global minimum target (converted to minutes)
        target_minutes = self.TARGET_WEEKLY_HOURS * 60
        self.model.Add(sum(all_active_minutes) >= target_minutes)
        
        # CONSTRAINT 3: No overlap per day
        for day in self.days:
            day_intervals = []
            for key, interval in self.study_intervals.items():
                if key[2] == day:
                    day_intervals.append(interval)
                    
            for fixed_day, fixed_start, fixed_end, module_code in self.fixed_modules:
                if fixed_day == day:
                    fixed_int = self.model.NewIntervalVar(
                        fixed_start, fixed_end - fixed_start, fixed_end, 
                        f"fixed_{module_code}_{day}_{fixed_start}"
                    )
                    day_intervals.append(fixed_int)
                    
            if day_intervals:
                self.model.AddNoOverlap(day_intervals)
    
    def add_break_constraints(self):
        """Add constraints for breaks between study sessions and lectures"""
        
        # 1. Global Minimum Break between ANY two study sessions on the SAME day
        keys = list(self.active_vars.keys())
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                key1 = keys[i]
                key2 = keys[j]
                
                if key1[2] == key2[2]:
                    act1 = self.active_vars[key1]
                    act2 = self.active_vars[key2]
                    start1 = self.start_vars[key1]
                    end1 = self.end_vars[key1]
                    start2 = self.start_vars[key2]
                    end2 = self.end_vars[key2]
                    
                    first_var = self.model.NewBoolVar(f"first_{i}_{j}")
                    self.model.Add(end1 + self.MIN_BREAK <= start2).OnlyEnforceIf([act1, act2, first_var])
                    self.model.Add(end2 + self.MIN_BREAK <= start1).OnlyEnforceIf([act1, act2, first_var.Not()])
        
        # 2. CLASS BUFFER: Ensure a buffer before and after any fixed lecture
        for key, active_var in self.active_vars.items():
            start_var = self.start_vars[key]
            end_var = self.end_vars[key]
            day = key[2]
            
            for fixed_day, fixed_start, fixed_end, module_code in self.fixed_modules:
                if fixed_day == day:
                    lec_id = f"{module_code}_{fixed_start}"
                    b_before_lec = self.model.NewBoolVar(f"before_lec_{key}_{lec_id}")
                    b_after_lec = self.model.NewBoolVar(f"after_lec_{key}_{lec_id}")
                    
                    self.model.Add(end_var <= fixed_start - self.CLASS_BUFFER).OnlyEnforceIf(b_before_lec)
                    self.model.Add(start_var >= fixed_end + self.CLASS_BUFFER).OnlyEnforceIf(b_after_lec)
                    self.model.AddBoolOr([b_before_lec, b_after_lec]).OnlyEnforceIf(active_var)
            
            # ABSA handling for Fridays
            if day == 'Friday' and self.absa_dates:
                absa_start = 8*60 + 30
                absa_end = 10*60 + 30
                
                ends_before_absa = self.model.NewBoolVar(f"b_absa_{key}")
                starts_after_absa = self.model.NewBoolVar(f"a_absa_{key}")
                
                self.model.Add(end_var <= absa_start).OnlyEnforceIf(ends_before_absa)
                self.model.Add(start_var >= absa_end).OnlyEnforceIf(starts_after_absa)
                self.model.AddBoolOr([ends_before_absa, starts_after_absa]).OnlyEnforceIf(active_var)
    
    def add_objective(self):
        """Add objective function"""
        objective_terms = []
        for key, active_var in self.active_vars.items():
            start_var = self.start_vars[key]
            start_penalty = self.model.NewIntVar(0, self.DAY_END, f"penalty_{key}")
            self.model.Add(start_penalty == start_var - self.DAY_START).OnlyEnforceIf(active_var)
            self.model.Add(start_penalty == 0).OnlyEnforceIf(active_var.Not())
            objective_terms.append(start_penalty)
        self.model.Minimize(sum(objective_terms))
    
    def solve(self):
        """Solve the model"""
        print("="*70)
        print("STUDY SCHEDULE OPTIMIZER FOR BMI POSTGRADUATE STUDENTS")
        print("="*70)
        
        self.solver.parameters.max_time_in_seconds = 10.0
        status = self.solver.Solve(self.model)
        
        if status == cp_model.OPTIMAL:
            print("✅ Optimal solution found!")
        elif status == cp_model.FEASIBLE:
            print("⚠️  Feasible solution found (not necessarily optimal)")
        else:
            print(f"❌ No solution found. Status: {status}")
            return None
        
        self.solution_schedule = self._extract_solution()
        return self.solution_schedule
    
    def _extract_solution(self):
        """Extract the schedule from the solution"""
        schedule = []
        for key, active_var in self.active_vars.items():
            if self.solver.Value(active_var):
                module, session_num, day = key
                start = self.solver.Value(self.start_vars[key])
                end = self.solver.Value(self.end_vars[key])
                
                schedule.append({
                    'module': module,
                    'session_id': f"{module}_{session_num}",
                    'day': day,
                    'start': self._minutes_to_time(start),
                    'end': self._minutes_to_time(end),
                    'start_minutes': start,
                    'end_minutes': end,
                    'type': 'study'
                })
        
        for fixed_day, fixed_start, fixed_end, module_code in self.fixed_modules:
            schedule.append({
                'module': module_code,
                'day': fixed_day,
                'start': self._minutes_to_time(fixed_start),
                'end': self._minutes_to_time(fixed_end),
                'start_minutes': fixed_start,
                'end_minutes': fixed_end,
                'type': 'lecture'
            })
        
        day_order = {day: i for i, day in enumerate(self.days)}
        schedule.sort(key=lambda x: (day_order[x['day']], x['start_minutes']))
        return schedule
    
    def _minutes_to_time(self, minutes):
        hours = minutes // 60
        mins = minutes % 60
        return f"{hours:02d}:{mins:02d}"
    
    def print_schedule(self):
        if not self.solution_schedule:
            return
        
        print("\n" + "="*70)
        print("📅 OPTIMIZED STUDY SCHEDULE")
        print("="*70)
        
        current_day = None
        for session in sorted(self.solution_schedule, key=lambda x: (self.day_to_int[x['day']], x['start_minutes'])):
            if session['day'] != current_day:
                current_day = session['day']
                print(f"\n{current_day}:")
            
            label = session['module'].replace('LECTURE_', 'LEC ') if session['type'] == 'lecture' else session['module']
            print(f"  {session['start']}-{session['end']} | {label}")

    def create_weekly_timetable(self, figsize=(15, 8)):
        if not self.solution_schedule:
            return None
        
        fig, ax = plt.subplots(figsize=figsize)
        time_slots = np.arange(self.DAY_START, self.DAY_END + 30, 30)
        time_labels = [self._minutes_to_time(t) for t in time_slots[::2]]
        
        for i, day in enumerate(self.days):
            for t in time_slots:
                ax.axhline(y=t, xmin=i/5, xmax=(i+1)/5, color='lightgray', linewidth=0.5, alpha=0.3)
            ax.axvline(x=i, color='black', linewidth=1)
        
        for session in self.solution_schedule:
            day_idx = self.day_to_int[session['day']]
            start = session['start_minutes']
            end = session['end_minutes']
            duration = end - start
            
            if session['type'] == 'lecture':
                color = self.module_colors.get('LECTURE', '#E0E0E0')
                alpha = 0.5
            else:
                module = session['module']
                color = self.module_colors.get(module, '#999999')
                alpha = 0.8
            
            rect = plt.Rectangle(
                (day_idx - 0.45, start), 0.9, duration,
                facecolor=color, edgecolor='black', linewidth=1, alpha=alpha
            )
            ax.add_patch(rect)
            
            if duration >= 45:
                label = session['module'].replace('LECTURE_', 'LEC ') if session['type'] == 'lecture' else session['module']
                ax.text(
                    day_idx, start + duration/2, label,
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    color='black' if session['type'] == 'lecture' else 'white'
                )
        
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(self.DAY_END, self.DAY_START)
        ax.set_xticks(range(5))
        ax.set_xticklabels(self.days, fontsize=11, fontweight='bold')
        ax.set_yticks(time_slots[::2])
        ax.set_yticklabels(time_labels)
        ax.set_ylabel('Time', fontsize=11)
        ax.set_title(f'Weekly Study Schedule ({self.TARGET_WEEKLY_HOURS} Hours Target)', fontsize=14, fontweight='bold', pad=20)
        
        legend_elements = [mpatches.Patch(color=c, label=m) for m, c in self.module_colors.items() if m not in ['BREAK', 'LECTURE']]
        legend_elements.append(mpatches.Patch(color=self.module_colors['LECTURE'], alpha=0.5, label='Fixed Lectures'))
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.01, 1))
        
        plt.tight_layout()
        return fig

    def show_visualizations(self):
        print("\n" + "="*70)
        print("🎨 GENERATING VISUALIZATIONS")
        print("="*70)
        
        fig1 = self.create_weekly_timetable()
        if fig1:
            save_path = 'bmi_weekly_timetable.png'
            fig1.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Timetable successfully saved as '{save_path}'")
            
        print("\n✅ Visualizations complete!")

# ============================================
# MAIN EXECUTION
# ============================================

def main():
    scheduler = StudyScheduler()
    scheduler.create_model()
    scheduler.add_hard_constraints()
    scheduler.add_break_constraints()
    scheduler.add_objective()
    
    schedule = scheduler.solve()
    
    if schedule:
        scheduler.print_schedule()
        scheduler.show_visualizations()
    else:
        print("\n❌ Could not create a valid schedule.")

if __name__ == "__main__":
    main()