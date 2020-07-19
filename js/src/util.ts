export function repeatFloat(value: number, nitems: number): Float32Array {
    const array = new Float32Array(nitems)
    array.fill(value)
    return array
}

export function linspace(min: number, max: number, n: number): Float32Array {
    const n1 = n - 1
    const step = (max - min) / n1
    const array = new Float32Array(n)
    for (let i = 0; i < n; i += 1) {
        array[i] = min + i * step
    }
    return array
}

export function createFloatArrayXyz(
    x: Array<number>, y: Array<number>, z: Array<number>,
): Float32Array {
    const n = Math.max(x.length, y.length, z.length)
    let i3 = 0
    const array = new Float32Array(3 * n)
    for (let i = 0; i < n; i += 1) {
        array[i3] = x[i]
        array[i3 + 1] = y[i]
        array[i3 + 2] = z[i]
        i3 += 3
    }
    return array
}

export function lightenColor(color: number): number {
    const light = 76
    let R = (color >> 16)
    let G = ((color >> 8) & 0x00FF)
    let B = (color & 0x0000FF)
    R = (R === 0) ? 110 : R + light
    G = (G === 0) ? 110 : G + light
    B = (B === 0) ? 111 : B + light
    R = Math.min(Math.max(R, 0), 255)
    G = Math.min(Math.max(G, 0), 255)
    B = Math.min(Math.max(B, 0), 255)
    return (0x1000000 + R * 0x10000 + G * 0x100 + B)
}
