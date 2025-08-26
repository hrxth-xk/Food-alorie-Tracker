"use client";

import React, { useEffect, useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Plus, Trash2, Salad, Flame, Settings2 } from "lucide-react";

// --- Types
type Macro = { calories: number; protein: number; carbs: number; fat: number };
type Food = { id: string; name: string; serving: string; macros: Macro };
type Entry = { id: string; foodId: string; name: string; quantity: number; macros: Macro; when: string };
type Goals = { calories: number; protein: number; carbs: number; fat: number };

// --- Sample embedded food DB (per 100g or standard serving)
const FOOD_DB: Food[] = [
  { id: "1", name: "Boiled Egg", serving: "1 egg (50g)", macros: { calories: 78, protein: 6, carbs: 0.6, fat: 5 } },
  { id: "2", name: "Chicken Breast (cooked)", serving: "100 g", macros: { calories: 165, protein: 31, carbs: 0, fat: 3.6 } },
  { id: "3", name: "Rice (cooked)", serving: "1 cup (158 g)", macros: { calories: 206, protein: 4.3, carbs: 45, fat: 0.4 } },
  { id: "4", name: "Chapati (roti)", serving: "1 piece (40 g)", macros: { calories: 120, protein: 3.1, carbs: 18, fat: 3.7 } },
  { id: "5", name: "Paneer", serving: "100 g", macros: { calories: 265, protein: 18, carbs: 6, fat: 20 } },
  { id: "6", name: "Dal (lentil curry)", serving: "1 cup (198 g)", macros: { calories: 230, protein: 18, carbs: 39, fat: 0.8 } },
  { id: "7", name: "Curd (plain yogurt)", serving: "100 g", macros: { calories: 61, protein: 3.5, carbs: 4.7, fat: 3.3 } },
  { id: "8", name: "Banana", serving: "1 medium (118 g)", macros: { calories: 105, protein: 1.3, carbs: 27, fat: 0.4 } },
  { id: "9", name: "Peanut Butter", serving: "1 tbsp (16 g)", macros: { calories: 94, protein: 3.6, carbs: 3.2, fat: 8 } },
  { id: "10", name: "Milk (toned)", serving: "200 ml", macros: { calories: 102, protein: 6.8, carbs: 9.8, fat: 3.2 } },
];

// --- Helpers
const uuid = () => Math.random().toString(36).slice(2);
const sumMacros = (items: { macros: Macro }[]) =>
  items.reduce(
    (acc, x) => ({
      calories: acc.calories + x.macros.calories,
      protein: acc.protein + x.macros.protein,
      carbs: acc.carbs + x.macros.carbs,
      fat: acc.fat + x.macros.fat,
    }),
    { calories: 0, protein: 0, carbs: 0, fat: 0 }
  );

const clamp = (n: number, min = 0) => (n < min ? min : n);

// --- Local storage hooks
const useLocal = <T,>(key: string, initial: T) => {
  const [state, setState] = useState<T>(() => {
    if (typeof window === "undefined") return initial;
    const raw = localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : initial;
  });
  useEffect(() => {
    if (typeof window !== "undefined") {
      localStorage.setItem(key, JSON.stringify(state));
    }
  }, [key, state]);
  return [state, setState] as const;
};

// --- Suggestion engine (simple rule-based)
function buildSuggestions(consumed: Macro, goals: Goals) {
  const remaining = {
    calories: clamp(goals.calories - consumed.calories),
    protein: clamp(goals.protein - consumed.protein),
    carbs: clamp(goals.carbs - consumed.carbs),
    fat: clamp(goals.fat - consumed.fat),
  };
  const tips: string[] = [];

  if (remaining.protein > 25) tips.push("You're low on protein. Add 150g chicken breast or 100g paneer.");
  if (remaining.carbs > 50) tips.push("Consider 1 cup cooked rice or 2 rotis to meet your carb goal.");
  if (remaining.fat > 15) tips.push("Healthy fats: 1 tbsp peanut butter or a handful of nuts.");
  if (consumed.calories > goals.calories) tips.push("You've exceeded calories. Go for a light dinner and add a 20–30 min walk.");
  if (consumed.carbs / Math.max(consumed.protein, 1) > 5) tips.push("Carb-heavy day. Balance with high-protein snacks (curd, eggs).");
  if (!tips.length) tips.push("Nice balance so far. Keep portions steady and hydrate well.");
  return { remaining, tips };
}

// --- Main component
export default function CalorieTrackerMVP() {
  const todayKey = new Date().toISOString().slice(0, 10);
  const [entries, setEntries] = useLocal<Entry[]>(`entries:${todayKey}`, []);
  const [goals, setGoals] = useLocal<Goals>("goals", { calories: 2200, protein: 130, carbs: 250, fat: 70 });
  const [q, setQ] = useState("");
  const [qty, setQty] = useState(1);
  const [selected, setSelected] = useState<Food | null>(null);
  const [showSettings, setShowSettings] = useState(false);

  const filtered = useMemo(() => FOOD_DB.filter(f => f.name.toLowerCase().includes(q.toLowerCase())), [q]);

  const consumed = useMemo(() => sumMacros(entries), [entries]);
  const { remaining, tips } = useMemo(() => buildSuggestions(consumed, goals), [consumed, goals]);

  function addSelected() {
    if (!selected) return;
    const m = selected.macros;
    const scaled: Macro = {
      calories: +(m.calories * qty).toFixed(1),
      protein: +(m.protein * qty).toFixed(1),
      carbs: +(m.carbs * qty).toFixed(1),
      fat: +(m.fat * qty).toFixed(1),
    };
    const e: Entry = {
      id: uuid(),
      foodId: selected.id,
      name: selected.name,
      when: new Date().toISOString(),
      quantity: qty,
      macros: scaled,
    };
    setEntries([...entries, e]);
    setSelected(null);
    setQty(1);
    setQ("");
  }

  function removeEntry(id: string) {
    setEntries(entries.filter(e => e.id !== id));
  }

  const progress = (v: number, g: number) => Math.min(100, Math.round((v / Math.max(g, 1)) * 100));

  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100 p-6">
      <div className="max-w-5xl mx-auto">
        <header className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-3">
            <Salad className="w-8 h-8" />
            <h1 className="text-2xl font-semibold">Food & Calorie Tracker</h1>
          </div>
          <button
            onClick={() => setShowSettings(s => !s)}
            className="inline-flex items-center gap-2 rounded-2xl px-4 py-2 bg-neutral-800 hover:bg-neutral-700"
          >
            <Settings2 className="w-4 h-4" /> Goals
          </button>
        </header>

        {/* Goals Panel */}
        {showSettings && (
          <motion.div initial={{ opacity: 0, y: -8 }} animate={{ opacity: 1, y: 0 }} className="grid md:grid-cols-4 gap-3 mb-6">
            {(["calories", "protein", "carbs", "fat"] as (keyof Goals)[]).map(k => (
              <div key={k} className="bg-neutral-900 rounded-2xl p-4">
                <label className="text-sm capitalize opacity-75">{k}</label>
                <input
                  type="number"
                  className="mt-2 w-full bg-neutral-800 rounded-xl px-3 py-2"
                  value={goals[k]}
                  onChange={e => setGoals({ ...goals, [k]: Number(e.target.value) })}
                />
              </div>
            ))}
          </motion.div>
        )}

        {/* Dashboard */}
        <div className="grid md:grid-cols-4 gap-4 mb-6">
          <StatCard
            title="Calories"
            value={`${consumed.calories.toFixed(0)} / ${goals.calories}`}
            percent={progress(consumed.calories, goals.calories)}
            icon={<Flame className="w-5 h-5" />}
          />
          <StatCard
            title="Protein (g)"
            value={`${consumed.protein.toFixed(0)} / ${goals.protein}`}
            percent={progress(consumed.protein, goals.protein)}
          />
          <StatCard
            title="Carbs (g)"
            value={`${consumed.carbs.toFixed(0)} / ${goals.carbs}`}
            percent={progress(consumed.carbs, goals.carbs)}
          />
          <StatCard
            title="Fat (g)"
            value={`${consumed.fat.toFixed(0)} / ${goals.fat}`}
            percent={progress(consumed.fat, goals.fat)}
          />
        </div>

        {/* Add food */}
        <div className="bg-neutral-900 rounded-2xl p-4 mb-6">
          <div className="flex flex-col md:flex-row gap-3 items-stretch md:items-end">
            <div className="flex-1">
              <label className="text-sm opacity-70">Search food</label>
              <input
                value={q}
                onChange={e => setQ(e.target.value)}
                placeholder="e.g., rice, roti, egg"
                className="w-full mt-2 bg-neutral-800 rounded-xl px-3 py-2"
              />
            </div>
            <div className="md:w-40">
              <label className="text-sm opacity-70">Qty (× serving)</label>
              <input
                type="number"
                min={0.25}
                step={0.25}
                value={qty}
                onChange={e => setQty(Number(e.target.value))}
                className="w-full mt-2 bg-neutral-800 rounded-xl px-3 py-2"
              />
            </div>
            <button
              disabled={!selected}
              onClick={addSelected}
              className={`inline-flex items-center justify-center gap-2 rounded-2xl px-4 py-3 ${
                selected ? "bg-emerald-600 hover:bg-emerald-500" : "bg-neutral-700 opacity-60"
              }`}
            >
              <Plus className="w-4 h-4" /> Add
            </button>
          </div>

          {/* Results */}
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-3 mt-4">
            {filtered.map(f => (
              <button
                key={f.id}
                onClick={() => setSelected(f)}
                className={`text-left bg-neutral-800 rounded-2xl p-3 hover:bg-neutral-700 border ${
                  selected?.id === f.id ? "border-emerald-500" : "border-transparent"
                }`}
              >
                <div className="font-medium">{f.name}</div>
                <div className="text-xs opacity-70">{f.serving}</div>
                <div className="text-xs mt-1 opacity-90">
                  {f.macros.calories} kcal • P {f.macros.protein}g • C {f.macros.carbs}g • F {f.macros.fat}g
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Entries */}
        <div className="bg-neutral-900 rounded-2xl p-4">
          <div className="flex items-center justify-between mb-2">
            <h3 className="font-semibold">Today's log</h3>
            <div className="text-sm opacity-70">{new Date().toDateString()}</div>
          </div>
          {entries.length === 0 ? (
            <div className="text-sm opacity-70">No entries yet. Search above to add your first food.</div>
          ) : (
            <ul className="divide-y divide-neutral-800">
              {entries.map(e => (
                <li key={e.id} className="py-3 flex items-center justify-between">
                  <div>
                    <div className="font-medium">
                      {e.name} × {e.quantity}
                    </div>
                    <div className="text-xs opacity-75">
                      {new Date(e.when).toLocaleTimeString()} • {e.macros.calories} kcal • P {e.macros.protein}g • C {e.macros.carbs}g • F {e.macros.fat}g
                    </div>
                  </div>
                  <button onClick={() => removeEntry(e.id)} className="rounded-xl p-2 bg-neutral-800 hover:bg-neutral-700">
                    <Trash2 className="w-4 h-4" />
                  </button>
                </li>
              ))}
            </ul>
          )}
        </div>

        {/* Suggestions */}
        <div className="grid md:grid-cols-2 gap-4 mt-6">
          <div className="bg-neutral-900 rounded-2xl p-4">
            <h3 className="font-semibold mb-2">Remaining today</h3>
            <div className="text-sm opacity-90">
              {remaining.calories} kcal • P {remaining.protein}g • C {remaining.carbs}g • F {remaining.fat}g
            </div>
          </div>
          <div className="bg-neutral-900 rounded-2xl p-4">
            <h3 className="font-semibold mb-2">Suggestions</h3>
            <ul className="list-disc pl-5 space-y-1 text-sm">
              {tips.map((t, i) => (
                <li key={i}>{t}</li>
              ))}
            </ul>
          </div>
        </div>

        <footer className="mt-10 text-xs opacity-60">
          Data is stored locally for this MVP. Set your goals from the top-right. Add real DB & auth in production.
        </footer>
      </div>
    </div>
  );
}

function StatCard({ title, value, percent, icon }: { title: string; value: string; percent: number; icon?: React.ReactNode }) {
  return (
    <div className="bg-neutral-900 rounded-2xl p-4">
      <div className="flex items-center justify-between">
        <div className="text-sm opacity-75">{title}</div>
        {icon}
      </div>
      <div className="text-xl font-semibold mt-1">{value}</div>
      <div className="h-2 bg-neutral-800 rounded-full mt-3 overflow-hidden">
        <div className="h-full" style={{ backgroundColor: "#10b981", width: `${percent}%` }} />
      </div>
    </div>
  );
}
