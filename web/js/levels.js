// levels.py -- the three theatres and their launch keys.
import { WIDTH, HEIGHT, HUD_HEIGHT, MISSILE_COLOR, BALLISTIC_COLOR } from "./config.js";
import { L3_LEAD } from "./doctrine.js";

const GROUND_Y = HEIGHT - HUD_HEIGHT;

export const LEVELS = [
  {
    number: 1, name: "OPEN SKY", tagline: "the original arena",
    blurb: ["Homing battery west, ballistic east.", "Missiles flank, weave and dart."],
    obstacles: [],
    stations: [
      { kind: "homing", x: 100, y: GROUND_Y, key: "H", label: "HOMING", color: MISSILE_COLOR, cluster_key: "C" },
      { kind: "ballistic", x: WIDTH - 100, y: GROUND_Y, key: "B", label: "BALLISTIC", color: BALLISTIC_COLOR, cluster_key: null },
    ],
    smart: false, datalink: false, lead: 0.0, pincer: false,
    legend: [["H", "homing missile"], ["C", "cluster of five"], ["B", "ballistic shot"]],
  },
  {
    number: 2, name: "TERRAIN MASKING", tagline: "terrain blocks line of sight",
    blurb: ["Peaks block radar; seekers fly around.", "Lose the lock: it flies to your last spot."],
    obstacles: [[230, 140, 40], [570, 140, 40], [400, 200, 32], [230, 360, 34], [560, 360, 38]],
    stations: [
      { kind: "homing", x: 100, y: GROUND_Y, key: "H", label: "SEEKER", color: MISSILE_COLOR, cluster_key: "C" },
      { kind: "ballistic", x: WIDTH - 100, y: GROUND_Y, key: "B", label: "BALLISTIC", color: BALLISTIC_COLOR, cluster_key: null },
    ],
    smart: true, datalink: false, lead: 0.0, pincer: false,
    legend: [["H", "seeker missile"], ["C", "seeker cluster"], ["B", "ballistic shot"]],
  },
  {
    number: 3, name: "PINCER NETWORK", tagline: "two homing batteries, datalinked salvos",
    blurb: ["One lock guides every missile in its salvo.", "Lead pursuit and pincers cut off escape."],
    obstacles: [],
    stations: [
      { kind: "homing", x: 100, y: GROUND_Y, key: "H", label: "WEST", color: MISSILE_COLOR, cluster_key: "C" },
      { kind: "homing", x: WIDTH - 100, y: GROUND_Y, key: "K", label: "EAST", color: MISSILE_COLOR, cluster_key: "M" },
    ],
    smart: true, datalink: true, lead: L3_LEAD, pincer: true,
    legend: [["H", "west missile"], ["C", "west cluster"], ["K", "east missile"],
             ["M", "east cluster"], ["S", "pincer salvo"]],
  },
];
