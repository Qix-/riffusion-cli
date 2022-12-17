#!/usr/bin/env -S node --no-warnings
import {promises as fsp} from 'node:fs';
import path from 'node:path';
import {spawnSync} from 'node:child_process';

import arg from 'arg';

const args = Object.assign({
	'--outdir': path.resolve('./samples'),
	'--seed': '?',
	'--seed-img': 'og-beat',
	'--guidance': '?',
	'--denoise': '?',
	'--inference-steps': 50,
	'--steps': 10
}, arg({
	'--outdir': String,
	'-o': '--outdir',

	'--seed': String,
	'-s': '--seed',

	'--seed-img': String,
	'-S': '--seed-img',

	'--guidance': String,
	'-g': '--guidance',

	'--denoise': String,
	'-d': '--denoise',

	'--inference-steps': Number,
	'-i': '--inference-steps',

	'--steps': Number,
	'-n': '--steps'
}, {
	stopAtPositional: true
}));

const seedImages = new Map([
	['og-beat', 'og_beat'],
	['agile', 'agile'],
	['marim', 'marim'],
	['motorway', 'motorway'],
	['vibes', 'vibes']
]);

const checkRange = (label, v, low, high) => {
	if (v < low || v > high) {
		throw new Error(`invalid ${label} range: ${low} <= ${v} <= ${high}`);
	}
}

args['--steps'] = Math.floor(args['--steps']);
args['--inference-steps'] = Math.floor(args['--inference-steps']);

if (args._.length === 0) throw new Error('missing prompt');
if (!seedImages.has(args['--seed-img'])) throw new Error(`invalid seed image: ${args['--seed-img']}`);
if (args['--steps'] < 1) throw new Error(`--steps must be >=1`);
if (args['--inference-steps'] < 1) throw new Error(`--inference-steps must be >=1`);

const breakIndex = args._.indexOf('--');
let startPrompt, endPrompt;
if (breakIndex === -1) {
	startPrompt = endPrompt = args._;
} else {
	startPrompt = args._.slice(0, breakIndex);
	endPrompt = args._.slice(breakIndex + 1);
}

startPrompt = startPrompt.join(' ').trim();
endPrompt = endPrompt.join(' ').trim();

if (startPrompt.length === 0) throw new Error('start prompt (before `--`) cannot be empty');
if (endPrompt.length === 0) throw new Error('end prompt (after `--`) cannot be empty');

const randInt = () => Math.floor(Math.random() * 999999);
const parseInt10 = n => parseInt(n, 10);
function parseRangeWithDefaults(arg, parseFn, def) {
	const matches = args[arg].match(/^(\+\d+)|(?:(\?|\d+)(?:-(\?|\d+))?)$/);
	if (!matches) throw new Error(`invalid ${arg} range: ${args[arg]}`);
	return matches[1] 
	? [ def, def + parseFn(matches[1]) ]
	: [
		matches[2] === '?' ? def : parseFn(matches[2]),
		(matches[3] ?? '?') === '?' ? def : parseFn(matches[3])
	];
}

let [startSeed, endSeed] = parseRangeWithDefaults('--seed', parseInt10, randInt());
const [startGuidance, endGuidance] = parseRangeWithDefaults('--guidance', parseInt10, 7);
const [startDenoise, endDenoise] = parseRangeWithDefaults('--denoise', parseFloat, 0.75);

checkRange('(start) denoise', startDenoise, 0, 1);
checkRange('end denoise', endDenoise, 0, 1);

const l = (t, s, e, d = '-') => s === e ? `${t}${s}` : `${t}${s}${d}${e}`;

console.error('to reproduce this result:');
console.error(
	'    riffusion',
	'-s', l('', startSeed, endSeed),
	'-S', args['--seed-img'],
	'-g', l('', startGuidance, endGuidance),
	'-d', l('', startDenoise, endDenoise),
	'-i', args['--inference-steps'],
	'-n', args['--steps'],
	l('', startPrompt, endPrompt, ' -- ')
);


const sleep = ms => new Promise(r => setTimeout(r, ms));

await fsp.mkdir(args['--outdir']).catch(error => {
	if (error?.code !== 'EEXIST') {
		throw error;
	}
});

const basePath = path.join(
	args['--outdir'],
	`${l('s', startSeed, endSeed)
	}_${args['--seed-img']
	}_${l('g', startGuidance, endGuidance)
	}_${l('d', startDenoise, endDenoise)
	}_i${args['--inference-steps']
	}_n${args['--steps']
	}__${l(
		'',
		startPrompt.replace(/\s+/g, '-'),
		endPrompt.replace(/\s+/g, '-'),
		'__'
	)}`);

await fsp.mkdir(basePath);

console.error('outputting to:', basePath);

if (startSeed === endSeed) ++endSeed;

const alphaStep = args['--steps'] === 1 ? Infinity : 1.0 / Math.max(1, args['--steps'] - 1);
const allSamples = [];
for (let alpha = 0; alpha <= 1.0; alpha += alphaStep) {
	console.error(`${(alpha * 100).toFixed(2)}% ...`);

	const requestPayload = Buffer.from(JSON.stringify({
		worklet_input: {
			alpha,
			mask_image_id: null,
			num_inference_steps: Math.floor(args['--inference-steps']),
			seed_image_id: seedImages.get(args['--seed-img']),
			start: {
				denoising: startDenoise,
				guidance: startGuidance,
				seed: startSeed,
				prompt: startPrompt
			},
			end: {
				denoising: endDenoise,
				guidance: endGuidance,
				seed: endSeed,
				prompt: endPrompt
			}
		}
	}), 'utf-8');

	for (let i = 1;;i++) {
		let response = await fetch('https://www.riffusion.com/api/baseten', {
			method: 'POST',
			headers: {
				'User-Agent': "A personal script I wrote, I'll try not to abuse it <3 Thanks for making this. tw @bad_at_computer if you want to yell at me :)",
				"Accept": "*/*",
				"Accept-Language": "en-US,en;q=0.9,de-DE;q=0.8,de;q=0.7",
				"Cache-Control": "no-cache",
				"Content-Type": "text/plain;charset=UTF-8",
				"Pragma": "no-cache",
			},
			body: requestPayload
		});
	
		if (response.status !== 200) {
			console.error(`WARNING: received non-200 response at alpha ${alpha}; got ${response.status}. Retry ${i} in 5s...`);
			await sleep(5000);
			continue;
		}

		const data = await response.json();

		if (!data.data.success) {
			console.error(`WARNING: model returned failure at alpha ${alpha}. Giving up!`);
			console.error(data);
			process.exit(1);
		}

		const latency = data?.data?.latency_ms || 'unknown';
		const modelOutput = data?.data?.worklet_output?.model_output;

		if (!modelOutput) {
			console.error(`WARNING: model_output unavailable at alpha ${alpha}. Retry ${i} in 5s...`);
			await sleep(5000);
			continue;
		}

		let modelData;
		try {
			modelData = JSON.parse(modelOutput);
		} catch (error) {
			console.error(`WARNING: model_output could not be parsed at alpha ${alpha}. Retry ${i} in 5s...`);
			console.error(error.stack);
			await sleep(5000);
			continue;
		}

		let audioData = modelData?.audio;
		if (!audioData) {
			console.error(`WARNING: audio data unavailable at alpha ${alpha}. Retry ${i} in 5s...`);
			await sleep(5000);
			continue;
		}

		const matches = audioData.match(/^data:audio\/mpeg;base64,(.+)$/is);
		if (!matches) {
			console.error(`WARNING: audio data in unexpected format at alpha ${alpha}. Retry ${i} in 5s...`);
			await sleep(5000);
			continue;
		}

		const mp364 = matches[1];
		const mp3 = Buffer.from(mp364, 'base64');

		const pathname = path.join(basePath, `${alpha.toFixed(10)}-l${latency}.mp3`);

		await fsp.writeFile(pathname, mp3);
		allSamples.push(pathname);

		break;
	}
}

console.error('writing sample manifest...');
const manifestPath = path.join(basePath, 'samples.txt');
await fsp.writeFile(
	manifestPath,
	allSamples.map(p => path.resolve(p)).map(p => `file '${p}'`).join('\n') + '\n',
	'utf-8'
);

const combinedPath = basePath + '.wav';
console.error('combining audio:', combinedPath);

const ffmpegArgs = [
	'-f', 'concat',
	'-safe', '0',
	'-i', manifestPath,
	'-c', 'copy',
	combinedPath
];

spawnSync(
	'ffmpeg',
	ffmpegArgs,
	{
		windowsHide: true,
		windowsVerbatimArguments: true,
		stdio: ['ignore', 'inherit', 'inherit']
	}
);
