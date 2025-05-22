# tsp_gui.py
import os
import time
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import openrouteservice
import tkintermapview
from geopy.geocoders import Nominatim
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# --- PSO + 2-opt Implementation ---
def calculate_length(tour, matrix):
    # Closed-loop tour length
    return sum(matrix[tour[i]][tour[(i+1) % len(tour)]] for i in range(len(tour)))

def calculate_swap_delta(tour, matrix, i, j):
    a, b = tour[i-1], tour[i]
    c, d = tour[j], tour[(j+1) % len(tour)]
    return (matrix[a][c] + matrix[b][d]) - (matrix[a][b] + matrix[c][d])

class HybridPSO2opt:
    def __init__(self, matrix, num_particles=50, max_iter=100, timeout=60,
                 w=0.5, c1=1.5, c2=1.5, improvement_threshold=0.02):
        self.matrix = matrix
        self.n = matrix.shape[0]
        self.max_iter = max_iter
        self.timeout = timeout
        self.w, self.c1, self.c2 = w, c1, c2
        self.improvement_threshold = improvement_threshold
        self.particles = []
        for _ in range(num_particles):
            pos = np.random.permutation(self.n)
            fit = calculate_length(pos, matrix)
            self.particles.append({'position': pos.copy(), 'best_position': pos.copy(), 'best_fitness': fit})
        best = min(self.particles, key=lambda x: x['best_fitness'])
        self.g_best, self.g_best_val = best['best_position'].copy(), best['best_fitness']

    def _get_segment(self, perm, size):
        start = np.random.randint(0, self.n - size + 1)
        return perm[start:start+size]

    def update_particle(self, p):
        i_size = max(2, int(self.n * self.w * 0.4))
        c_size = max(2, int(self.n * self.c1 * 0.3))
        s_size = max(2, int(self.n * self.c2 * 0.3))
        segs = [
            self._get_segment(p['position'], i_size),
            self._get_segment(p['best_position'], c_size),
            self._get_segment(self.g_best, s_size)
        ]
        np.random.shuffle(segs)
        new = []
        used = set()
        for seg in segs:
            for city in seg:
                if city not in used:
                    new.append(city)
                    used.add(city)
        missing = [c for c in range(self.n) if c not in used]
        if missing:
            current = new[-1] if new else np.random.choice(missing)
            while missing:
                nxt = min(missing, key=lambda x: self.matrix[current][x])
                new.append(nxt)
                missing.remove(nxt)
                current = nxt
        return np.array(new)

    def _apply_2opt(self, tour):
        best = tour.copy()
        improved = True
        while improved:
            improved = False
            for i in range(1, self.n-2):
                for j in range(i+1, self.n):
                    if calculate_swap_delta(best, self.matrix, i, j) < 0:
                        best[i:j+1] = best[i:j+1][::-1]
                        improved = True
        return best

    def _apply_2opt_to_best(self, top_k=3):
        for p in sorted(self.particles, key=lambda x: x['best_fitness'])[:top_k]:
            new_tour = self._apply_2opt(p['position'])
            new_fit = calculate_length(new_tour, self.matrix)
            if new_fit < p['best_fitness']:
                p.update({'position': new_tour.copy(), 'best_position': new_tour.copy(), 'best_fitness': new_fit})
                if new_fit < self.g_best_val:
                    self.g_best, self.g_best_val = new_tour.copy(), new_fit

    def run(self):
        start = time.time()
        prev_best = float('inf')
        last_imp = start
        for it in range(self.max_iter):
            if time.time() - start > self.timeout:
                break
            improved_any = False
            for p in self.particles:
                newp = self.update_particle(p)
                f = calculate_length(newp, self.matrix)
                if f < p['best_fitness']:
                    p['best_position'], p['best_fitness'], improved_any = newp.copy(), f, True
                p['position'] = newp
                if f < self.g_best_val:
                    self.g_best, self.g_best_val, improved_any = newp.copy(), f, True
            if it % 5 == 0:
                self._apply_2opt_to_best()
            if self.g_best_val < prev_best:
                imp = 1.0 if prev_best == float('inf') else (prev_best - self.g_best_val)/prev_best
                prev_best, last_imp = self.g_best_val, time.time()
                if imp < self.improvement_threshold:
                    self._apply_2opt_to_best(top_k=5)
            elif time.time() - last_imp > 5 and not improved_any:
                break
        return self.g_best_val, time.time() - start

# --- TSPLib Loader ---
def load_tsp_file(path):
    coords = []
    with open(path) as f:
        for line in f:
            if line.strip().startswith('NODE_COORD_SECTION'):
                break
        for line in f:
            if line.strip() == 'EOF':
                break
            parts = line.split()
            if len(parts) >= 3:
                coords.append((float(parts[1]), float(parts[2])))
    n = len(coords)
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                mat[i][j] = np.hypot(coords[i][0] - coords[j][0], coords[i][1] - coords[j][1])
    return coords, mat

# --- GUI Application ---
class TSPApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('TSP Route Solver')
        self.geometry('900x800')
        # API key must be set in .env
        self.api_key = os.getenv('ORS_API_KEY')
        if not self.api_key:
            messagebox.showerror('Error', 'ORS_API_KEY not set in .env')
            self.destroy(); return
        # State
        self.tsp_mode = False
        self.markers = []
        self.coords = []  # [lon, lat]
        self.current_path = None
        # Controls
        ctrl = tk.Frame(self)
        ctrl.pack(fill=tk.X, pady=5)
        tk.Button(ctrl, text='Load .tsp File', command=self.load_tsp).pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl, text='Export .tsp', command=self.export_tsp).pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl, text='Clear Points', command=self.clear_points).pack(side=tk.LEFT, padx=5)
        confirm_btn = tk.Button(ctrl, text='Confirm Route', command=self.confirm_route)
        confirm_btn.pack(side=tk.LEFT, padx=5)
        self.tsp_var = tk.BooleanVar(value=False)
        tk.Checkbutton(ctrl, text='TSP Mode', variable=self.tsp_var,
                       command=self.toggle_mode).pack(side=tk.LEFT, padx=5)
        self.search_entry = tk.Entry(ctrl, width=30)
        self.search_entry.pack(side=tk.LEFT, padx=5)
        tk.Button(ctrl, text='Search Place', command=self.search_place).pack(side=tk.LEFT, padx=5)
        # Map and output
        self.map_widget = tkintermapview.TkinterMapView(self, width=900, height=500)
        self.map_widget.set_position(39.92, 32.85)
        self.map_widget.set_zoom(6)
        self.map_widget.pack(fill=tk.BOTH, expand=True)
        self.map_widget.add_left_click_map_command(self.on_map_click)
        self.output = scrolledtext.ScrolledText(self, height=6, state='disabled')
        self.output.pack(fill=tk.X, padx=5)

    def toggle_mode(self):
        self.tsp_mode = self.tsp_var.get()

    def on_map_click(self, coord):
        # Add or remove waypoints: click exactly on marker removes it, else adds
        lat, lon = coord
        tol = 1e-5
        for m in list(self.markers):
            mlat, mlon = m.position
            if abs(mlat - lat) < tol and abs(mlon - lon) < tol:
                self.remove_marker(m)
                return
        self.add_marker(lat, lon)

    def add_marker(self, lat, lon):
        # Label starts from 0
        idx = len(self.markers)
        m = self.map_widget.set_marker(lat, lon, text=str(idx))
        self.markers.append(m)
        self.coords.append([lon, lat])

    def remove_marker(self, marker):
        idx = self.markers.index(marker)
        marker.delete()
        del self.markers[idx]
        del self.coords[idx]
        # Refresh labels
        for i, m in enumerate(self.markers):
            try:
                m.set_text(str(i))
            except:
                pass

    def clear_points(self):
        for m in self.markers:
            m.delete()
        if self.current_path:
            self.current_path.delete()
        self.markers.clear()
        self.coords.clear()
        self.current_path = None
        self.output.config(state='normal')
        self.output.delete('1.0', tk.END)
        self.output.config(state='disabled')

    def confirm_route(self):
        if len(self.coords) < 2:
            messagebox.showerror('Error', 'Select at least 2 points')
            return
        client = openrouteservice.Client(key=self.api_key)
        if self.tsp_mode:
            dm = client.distance_matrix(
                locations=self.coords,
                metrics=['distance'],
                sources=list(range(len(self.coords))),
                destinations=list(range(len(self.coords)))
            )
            matrix = np.array(dm['distances'])
            solver = HybridPSO2opt(matrix)
            dist, elapsed = solver.run()
            tour = solver.g_best.tolist() + [solver.g_best[0]]
            ordered = [self.coords[i] for i in tour]
            route = client.directions(ordered, profile='driving-car', format='geojson')
        else:
            ordered = self.coords + [self.coords[0]]
            route = client.directions(ordered, profile='driving-car', format='geojson')
            tour = list(range(len(self.coords))) + [0]
        segs = route['features'][0]['properties']['segments']
        total_dist = sum(s['distance'] for s in segs)
        total_dur = sum(s['duration'] for s in segs)
        geom = route['features'][0]['geometry']['coordinates']
        path = [(c[1], c[0]) for c in geom]
        if self.current_path:
            self.current_path.delete()
        self.current_path = self.map_widget.set_path(path)
        self.display_result(total_dist, total_dur, tour)

    def export_tsp(self):
        """
        Exports current waypoints to a .tsp file in proper lat lon order.
        """
        if not self.coords:
            messagebox.showerror('Error', 'No waypoints to export')
            return
        path = filedialog.asksaveasfilename(defaultextension='.tsp', filetypes=[('TSP File', '*.tsp')])
        if not path:
            return
        with open(path, 'w') as f:
            f.write(f"NAME: exported_route\nTYPE: TSP\nDIMENSION: {len(self.coords)}\nEDGE_WEIGHT_TYPE: EUC_2D\nNODE_COORD_SECTION\n")
            for i, (lon, lat) in enumerate(self.coords, start=1):
                # coords stored as [lon, lat], write as lat lon for TSPLIB
                f.write(f"{i} {lat:.6f} {lon:.6f}\n")
            f.write('EOF')
        messagebox.showinfo('Exported', f'Saved .tsp to {path}')

    def display_result(self, dist, elapsed, tour):
        self.output.config(state='normal')
        self.output.delete('1.0', tk.END)
        self.output.insert(tk.END, f"Distance: {dist:.1f} m\nTime: {elapsed:.2f} s\nTour: {tour}\n")
        self.output.config(state='disabled')

    def load_tsp(self):
        """
        Loads a .tsp file and places markers without auto-confirming.
        """
        path = filedialog.askopenfilename(filetypes=[('TSP Files', '*.tsp')])
        if not path:
            return
        coords, mat = load_tsp_file(path)
        # Clear existing points and paths
        self.clear_points()
        # Populate markers and self.coords ([lon, lat])
        for lat, lon in coords:
            m = self.map_widget.set_marker(lat, lon, text=str(len(self.markers)))
            self.markers.append(m)
            self.coords.append([lon, lat])
        # Switch to TSP mode and store matrix for later
        self.tsp_mode = True
        self.matrix_override = mat
        # Do NOT auto-run route solver; wait for Confirm Route button

    def search_place(self):
        q = self.search_entry.get().strip()
        if not q:
            return
        geo = Nominatim(user_agent='tsp_app')
        loc = geo.geocode(q)
        if not loc:
            messagebox.showinfo('Not found', f"Could not find '{q}'")
            return
        self.map_widget.set_position(loc.latitude, loc.longitude)
        self.add_marker(loc.latitude, loc.longitude)

if __name__ == '__main__':
    app = TSPApp()
    app.mainloop()
