// sim/geometry.py
export function segmentBlocked(ax, ay, bx, by, obstacles) {
  const dx = bx - ax, dy = by - ay, l2 = dx * dx + dy * dy;
  for (let i = 0; i < obstacles.length; i++) {
    const o = obstacles[i];
    const ex = o[0] - ax, ey = o[1] - ay;
    let t = 0.0;
    if (l2 > 1e-12) {
      t = (ex * dx + ey * dy) / l2;
      if (t < 0.0) t = 0.0; else if (t > 1.0) t = 1.0;
    }
    const cx = t * dx - ex, cy = t * dy - ey;
    if (Math.sqrt(cx * cx + cy * cy) < o[2]) return true;
  }
  return false;
}

export function insideAny(x, y, obstacles, pad = 0.0) {
  for (let i = 0; i < obstacles.length; i++) {
    const o = obstacles[i];
    if (Math.sqrt((x - o[0]) * (x - o[0]) + (y - o[1]) * (y - o[1])) < o[2] + pad) return true;
  }
  return false;
}

export function clearance(x, y, obstacles) {
  let best = Infinity;
  for (let i = 0; i < obstacles.length; i++) {
    const o = obstacles[i];
    best = Math.min(best, Math.sqrt((x - o[0]) * (x - o[0]) + (y - o[1]) * (y - o[1])) - o[2]);
  }
  return best;
}
