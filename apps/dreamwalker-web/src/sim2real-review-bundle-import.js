import { normalizeLocalizationEstimate, readNonEmptyString } from './sim2real-localization-core.js'

function isRecord(value) {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function parseRawDocument(rawDocument) {
  return typeof rawDocument === 'string' || rawDocument instanceof String
    ? JSON.parse(String(rawDocument))
    : rawDocument
}

function requireRecord(value, label) {
  if (!isRecord(value)) {
    throw new Error(`${label} must be a JSON object`)
  }

  return value
}

function firstRecord(record, keys) {
  for (const key of keys) {
    if (isRecord(record?.[key])) {
      return record[key]
    }
  }

  return null
}

function firstArray(record, keys) {
  for (const key of keys) {
    if (Array.isArray(record?.[key])) {
      return record[key]
    }
  }

  return null
}

function firstValue(record, keys) {
  for (const key of keys) {
    if (record?.[key] !== undefined && record[key] !== null) {
      return record[key]
    }
  }

  return undefined
}

function normalizeOptionalNumber(value) {
  const normalized = Number(value)
  return Number.isFinite(normalized) ? normalized : null
}

function normalizeCapturePose(poseLike) {
  const pose = requireRecord(poseLike, 'capture pose')
  const position = Array.isArray(pose.position) ? pose.position.slice(0, 3).map(Number) : null
  const yawDegrees = Number(pose.yawDegrees)

  if (!position || position.length !== 3 || position.some((value) => !Number.isFinite(value))) {
    throw new Error('capture pose.position must be three finite numbers')
  }

  if (!Number.isFinite(yawDegrees)) {
    throw new Error('capture pose.yawDegrees must be finite')
  }

  return {
    position,
    yawDegrees
  }
}

function normalizeCaptureResponse(responseLike) {
  const response = requireRecord(responseLike, 'capture response')

  if (readNonEmptyString(response.type) !== 'render-result') {
    throw new Error('capture response must be render-result')
  }

  if (!Number.isFinite(Number(response.width)) || !Number.isFinite(Number(response.height))) {
    throw new Error('capture response width/height are required')
  }

  if (!readNonEmptyString(response.colorJpegBase64) || !readNonEmptyString(response.depthBase64)) {
    throw new Error('capture response image payloads are required')
  }

  return {
    ...response,
    width: Number(response.width),
    height: Number(response.height)
  }
}

function normalizeRouteCaptureBundle(bundleLike) {
  const bundle = requireRecord(bundleLike, 'route capture bundle')

  if (readNonEmptyString(bundle.type) !== 'route-capture-bundle') {
    throw new Error('capture bundle type must be route-capture-bundle')
  }

  const captures = Array.isArray(bundle.captures)
    ? bundle.captures.map((entry, index) => {
        const capture = requireRecord(entry, `capture bundle entry ${index + 1}`)

        return {
          index: Number.isFinite(Number(capture.index)) ? Number(capture.index) : index,
          label: readNonEmptyString(capture.label) || `capture:${index + 1}`,
          capturedAt: readNonEmptyString(capture.capturedAt),
          relativeTimeSeconds: normalizeOptionalNumber(capture.relativeTimeSeconds),
          pose: normalizeCapturePose(capture.pose),
          response: normalizeCaptureResponse(capture.response)
        }
      })
    : []

  if (captures.length === 0) {
    throw new Error('capture bundle must include at least one capture')
  }

  return {
    protocol: readNonEmptyString(bundle.protocol) || 'dreamwalker-sim2real-capture/v1',
    type: 'route-capture-bundle',
    capturedAt: readNonEmptyString(bundle.capturedAt),
    fragmentId: readNonEmptyString(bundle.fragmentId),
    fragmentLabel: readNonEmptyString(bundle.fragmentLabel),
    endpoint: readNonEmptyString(bundle.endpoint),
    server: isRecord(bundle.server) ? bundle.server : {},
    request: isRecord(bundle.request) ? bundle.request : {},
    route: Array.isArray(bundle.route) ? bundle.route : [],
    captures
  }
}

function normalizeLocalizationImageBenchmarkHighlight(highlightLike) {
  const highlight = highlightLike && typeof highlightLike === 'object' ? highlightLike : {}
  const value = normalizeOptionalNumber(highlight.value)
  const frameIndex = Number.isFinite(Number(highlight.frameIndex))
    ? Number(highlight.frameIndex)
    : null

  if (!Number.isFinite(value) || frameIndex === null) {
    return null
  }

  return {
    ordering: readNonEmptyString(highlight.ordering) || 'max',
    frameIndex,
    value,
    groundTruthLabel: readNonEmptyString(highlight.groundTruthLabel),
    estimateLabel: readNonEmptyString(highlight.estimateLabel),
    interpolationKind: readNonEmptyString(highlight.interpolationKind),
    timeDeltaSeconds: normalizeOptionalNumber(highlight.timeDeltaSeconds),
    groundTruthColorJpegBase64: readNonEmptyString(highlight.groundTruthColorJpegBase64),
    renderedColorJpegBase64: readNonEmptyString(highlight.renderedColorJpegBase64)
  }
}

function normalizeLocalizationImageBenchmarkReport(reportLike) {
  const report = requireRecord(reportLike, 'image benchmark report')

  if (readNonEmptyString(report.type) !== 'localization-image-benchmark-report') {
    throw new Error('image benchmark report type must be localization-image-benchmark-report')
  }

  const summary =
    isRecord(report.metrics?.summary)
      ? report.metrics.summary
      : isRecord(report.summary)
        ? {
            psnr: { mean: report.summary.psnrMean },
            ssim: { mean: report.summary.ssimMean },
            lpips: { mean: report.summary.lpipsMean }
          }
        : {}
  const highlights =
    isRecord(report.metrics?.highlights)
      ? report.metrics.highlights
      : isRecord(report.highlights)
        ? report.highlights
        : {}

  return {
    protocol: readNonEmptyString(report.protocol) || 'dreamwalker-localization-image-benchmark/v1',
    type: 'localization-image-benchmark-report',
    createdAt: readNonEmptyString(report.createdAt),
    endpoint: readNonEmptyString(report.endpoint),
    alignment: readNonEmptyString(report.alignment) || 'auto',
    estimateLabel:
      readNonEmptyString(report.estimate?.label) ||
      readNonEmptyString(report.estimateLabel) ||
      'Localization Estimate',
    groundTruthLabel:
      readNonEmptyString(report.groundTruth?.label) ||
      readNonEmptyString(report.groundTruth?.fragmentLabel) ||
      readNonEmptyString(report.groundTruth?.fragmentId) ||
      'Ground Truth Capture',
    matchedCount: Number.isFinite(Number(report.matching?.matchedCount))
      ? Number(report.matching.matchedCount)
      : 0,
    summary: {
      psnrMean: normalizeOptionalNumber(summary.psnr?.mean),
      ssimMean: normalizeOptionalNumber(summary.ssim?.mean),
      lpipsMean: normalizeOptionalNumber(summary.lpips?.mean)
    },
    highlights: {
      psnr: normalizeLocalizationImageBenchmarkHighlight(highlights.psnr),
      ssim: normalizeLocalizationImageBenchmarkHighlight(highlights.ssim),
      lpips: normalizeLocalizationImageBenchmarkHighlight(highlights.lpips)
    }
  }
}

function buildSummaryFromReport(reportLike, estimateLabel, groundTruthLabel, requestedAlignment) {
  const report = isRecord(reportLike) ? reportLike : {}

  return {
    createdAt: readNonEmptyString(report.createdAt),
    alignment: readNonEmptyString(report.alignment) || requestedAlignment || 'auto',
    requestedAlignment:
      readNonEmptyString(report.requestedAlignment) ||
      readNonEmptyString(report.alignment) ||
      requestedAlignment ||
      'auto',
    groundTruthLabel:
      readNonEmptyString(report.groundTruth?.label) || groundTruthLabel || 'Ground Truth Capture',
    estimateLabel:
      readNonEmptyString(report.estimate?.label) || estimateLabel || 'Localization Estimate',
    sourceType:
      readNonEmptyString(report.estimate?.sourceType) || 'poses',
    interpolationMode:
      readNonEmptyString(report.estimate?.interpolationMode) || 'none',
    matchedCount: Number.isFinite(Number(report.matching?.matchedCount))
      ? Number(report.matching.matchedCount)
      : 0,
    groundTruthPoseCount: Number.isFinite(Number(report.groundTruth?.poseCount))
      ? Number(report.groundTruth.poseCount)
      : 0,
    estimatePoseCount: Number.isFinite(Number(report.estimate?.poseCount))
      ? Number(report.estimate.poseCount)
      : 0,
    ateRmseMeters: normalizeOptionalNumber(report.metrics?.ateRmseMeters),
    yawRmseDegrees: normalizeOptionalNumber(report.metrics?.yaw?.rmse),
    rpeTranslationRmseMeters: normalizeOptionalNumber(report.metrics?.rpeTranslationRmseMeters),
    rpeYawRmseDegrees: normalizeOptionalNumber(report.metrics?.rpeYawRmseDegrees),
    timeDeltaMeanSeconds: normalizeOptionalNumber(report.metrics?.timeDelta?.mean),
    timeDeltaMaxSeconds: normalizeOptionalNumber(report.metrics?.timeDelta?.max),
    timeAligned: Boolean(report.matching?.timeAligned),
    interpolatedCount: Number.isFinite(Number(report.matching?.interpolatedCount))
      ? Number(report.matching.interpolatedCount)
      : 0,
    clampedCount: Number.isFinite(Number(report.matching?.clampedCount))
      ? Number(report.matching.clampedCount)
      : 0
  }
}

function normalizeLocalizationRunSummary(summaryLike, fallbackSummary) {
  const summary = summaryLike && typeof summaryLike === 'object' ? summaryLike : {}

  return {
    createdAt: readNonEmptyString(summary.createdAt) || fallbackSummary.createdAt || '',
    alignment: readNonEmptyString(summary.alignment) || fallbackSummary.alignment || 'auto',
    requestedAlignment:
      readNonEmptyString(summary.requestedAlignment) ||
      fallbackSummary.requestedAlignment ||
      readNonEmptyString(summary.alignment) ||
      fallbackSummary.alignment ||
      'auto',
    groundTruthLabel:
      readNonEmptyString(summary.groundTruthLabel) ||
      fallbackSummary.groundTruthLabel ||
      'Ground Truth Capture',
    estimateLabel:
      readNonEmptyString(summary.estimateLabel) ||
      fallbackSummary.estimateLabel ||
      'Localization Estimate',
    sourceType:
      readNonEmptyString(summary.sourceType) ||
      fallbackSummary.sourceType ||
      'poses',
    interpolationMode:
      readNonEmptyString(summary.interpolationMode) ||
      fallbackSummary.interpolationMode ||
      'none',
    matchedCount: Number.isFinite(Number(summary.matchedCount))
      ? Number(summary.matchedCount)
      : Number(fallbackSummary.matchedCount || 0),
    groundTruthPoseCount: Number.isFinite(Number(summary.groundTruthPoseCount))
      ? Number(summary.groundTruthPoseCount)
      : Number(fallbackSummary.groundTruthPoseCount || 0),
    estimatePoseCount: Number.isFinite(Number(summary.estimatePoseCount))
      ? Number(summary.estimatePoseCount)
      : Number(fallbackSummary.estimatePoseCount || 0),
    ateRmseMeters: normalizeOptionalNumber(summary.ateRmseMeters ?? fallbackSummary.ateRmseMeters),
    yawRmseDegrees: normalizeOptionalNumber(
      summary.yawRmseDegrees ?? fallbackSummary.yawRmseDegrees
    ),
    rpeTranslationRmseMeters: normalizeOptionalNumber(
      summary.rpeTranslationRmseMeters ?? fallbackSummary.rpeTranslationRmseMeters
    ),
    rpeYawRmseDegrees: normalizeOptionalNumber(
      summary.rpeYawRmseDegrees ?? fallbackSummary.rpeYawRmseDegrees
    ),
    timeDeltaMeanSeconds: normalizeOptionalNumber(
      summary.timeDeltaMeanSeconds ?? fallbackSummary.timeDeltaMeanSeconds
    ),
    timeDeltaMaxSeconds: normalizeOptionalNumber(
      summary.timeDeltaMaxSeconds ?? fallbackSummary.timeDeltaMaxSeconds
    ),
    timeAligned:
      typeof summary.timeAligned === 'boolean'
        ? summary.timeAligned
        : Boolean(fallbackSummary.timeAligned),
    interpolatedCount: Number.isFinite(Number(summary.interpolatedCount))
      ? Number(summary.interpolatedCount)
      : Number(fallbackSummary.interpolatedCount || 0),
    clampedCount: Number.isFinite(Number(summary.clampedCount))
      ? Number(summary.clampedCount)
      : Number(fallbackSummary.clampedCount || 0)
  }
}

function normalizeCompareReport(reportLike) {
  if (!isRecord(reportLike)) {
    return null
  }

  if (readNonEmptyString(reportLike.type) !== 'localization-run-compare-report') {
    throw new Error('compare report type must be localization-run-compare-report')
  }

  return {
    ...reportLike,
    protocol: readNonEmptyString(reportLike.protocol) || 'dreamwalker-localization-run-compare/v1',
    type: 'localization-run-compare-report',
    createdAt: readNonEmptyString(reportLike.createdAt),
    baselineRunId: readNonEmptyString(reportLike.baselineRunId),
    baselineLabel: readNonEmptyString(reportLike.baselineLabel),
    rows: Array.isArray(reportLike.rows) ? reportLike.rows : []
  }
}

function deriveCaptureShelfEntryId(sourceId, createdAt, index) {
  if (sourceId.startsWith('capture-shelf:')) {
    return sourceId.slice('capture-shelf:'.length) || `review-capture-${index}`
  }

  const safeCreatedAt = readNonEmptyString(createdAt).replace(/[:.]/g, '-') || 'imported'
  return `review-capture-${safeCreatedAt}-${index}`
}

function normalizeLinkedCaptureEntry(entryLike, index, { aliases = false } = {}) {
  const entry = requireRecord(entryLike, `linked capture ${index + 1}`)
  const bundle = normalizeRouteCaptureBundle(
    firstValue(entry, aliases ? ['bundle', 'captureBundle'] : ['bundle'])
  )
  const sourceId =
    readNonEmptyString(firstValue(entry, aliases ? ['sourceId', 'captureSourceId'] : ['sourceId'])) ||
    `capture-shelf:${deriveCaptureShelfEntryId('', bundle.capturedAt, index)}`

  return {
    sourceId,
    label:
      readNonEmptyString(entry.label) ||
      bundle.fragmentLabel ||
      bundle.fragmentId ||
      `Capture ${index + 1}`,
    fragmentId: readNonEmptyString(entry.fragmentId) || bundle.fragmentId,
    fragmentLabel: readNonEmptyString(entry.fragmentLabel) || bundle.fragmentLabel,
    captureCount: Number.isFinite(Number(entry.captureCount))
      ? Number(entry.captureCount)
      : bundle.captures.length,
    bundle
  }
}

function normalizeRunSnapshot(
  snapshotLike,
  index,
  linkedCaptureMap,
  { aliases = false, allowLinkedCaptureFallback = false } = {}
) {
  const snapshot = requireRecord(snapshotLike, `review bundle snapshot ${index + 1}`)
  const estimate = normalizeLocalizationEstimate(
    firstValue(
      snapshot,
      aliases
        ? ['estimate', 'estimateInput', 'localizationEstimate', 'trajectory']
        : ['estimate', 'estimateInput', 'localizationEstimate']
    )
  )
  const groundTruth =
    firstRecord(snapshot, aliases ? ['groundTruth', 'groundTruthRef', 'captureSource'] : ['groundTruth']) || {}
  const sourceId =
    readNonEmptyString(
      firstValue(groundTruth, aliases ? ['sourceId', 'captureSourceId'] : ['sourceId'])
    ) || 'current-capture'
  let groundTruthBundle = null

  if (firstValue(groundTruth, aliases ? ['bundle', 'captureBundle'] : ['bundle'])) {
    groundTruthBundle = normalizeRouteCaptureBundle(
      firstValue(groundTruth, aliases ? ['bundle', 'captureBundle'] : ['bundle'])
    )
  } else if (allowLinkedCaptureFallback && linkedCaptureMap.has(sourceId)) {
    groundTruthBundle = linkedCaptureMap.get(sourceId)?.bundle ?? null
  } else {
    throw new Error(`review bundle snapshot ${index + 1} is missing groundTruth.bundle`)
  }

  const benchmark =
    firstRecord(snapshot, aliases ? ['benchmark', 'benchmarkConfig'] : ['benchmark']) || {}
  const report = snapshot.report && typeof snapshot.report === 'object' ? snapshot.report : null
  const fallbackSummary = buildSummaryFromReport(
    report,
    estimate.label,
    readNonEmptyString(groundTruth.label),
    readNonEmptyString(benchmark.requestedAlignment) || readNonEmptyString(benchmark.alignment)
  )
  const imageBenchmark = snapshot.imageBenchmark
    ? normalizeLocalizationImageBenchmarkReport(snapshot.imageBenchmark)
    : null

  return {
    protocol: readNonEmptyString(snapshot.protocol) || 'dreamwalker-localization-run/v1',
    type: 'localization-run-snapshot',
    id:
      readNonEmptyString(firstValue(snapshot, aliases ? ['id', 'runId'] : ['id'])) ||
      `review-run-${index}`,
    label:
      readNonEmptyString(firstValue(snapshot, aliases ? ['label', 'runLabel'] : ['label'])) ||
      estimate.label,
    savedAt: readNonEmptyString(snapshot.savedAt) || fallbackSummary.createdAt || '',
    groundTruth: {
      sourceId,
      label: readNonEmptyString(groundTruth.label) || fallbackSummary.groundTruthLabel,
      bundle: groundTruthBundle
    },
    estimate,
    benchmark: {
      alignment:
        readNonEmptyString(benchmark.alignment) ||
        readNonEmptyString(report?.alignment) ||
        fallbackSummary.alignment,
      requestedAlignment:
        readNonEmptyString(benchmark.requestedAlignment) ||
        readNonEmptyString(benchmark.alignment) ||
        readNonEmptyString(report?.requestedAlignment) ||
        readNonEmptyString(report?.alignment) ||
        fallbackSummary.requestedAlignment,
      reportCreatedAt:
        readNonEmptyString(benchmark.reportCreatedAt) ||
        readNonEmptyString(report?.createdAt) ||
        fallbackSummary.createdAt
    },
    summary: normalizeLocalizationRunSummary(snapshot.summary, fallbackSummary),
    report,
    imageBenchmark
  }
}

function normalizeRunEntry(
  runLike,
  index,
  linkedCaptureMap,
  { aliases = false, allowLinkedCaptureFallback = false } = {}
) {
  const run = requireRecord(runLike, `review bundle run ${index + 1}`)
  const snapshot =
    firstRecord(
      run,
      aliases ? ['snapshot', 'portableSnapshot', 'runSnapshot'] : ['snapshot']
    ) || run

  return {
    id:
      readNonEmptyString(firstValue(run, aliases ? ['id', 'runId'] : ['id'])) ||
      readNonEmptyString(firstValue(snapshot, aliases ? ['id', 'runId'] : ['id'])) ||
      `review-run-${index}`,
    label:
      readNonEmptyString(firstValue(run, aliases ? ['label', 'runLabel'] : ['label'])) ||
      readNonEmptyString(firstValue(snapshot, aliases ? ['label', 'runLabel'] : ['label'])) ||
      `Review Run ${index + 1}`,
    rank: Number.isFinite(Number(run.rank)) ? Number(run.rank) : index + 1,
    reviewArtifacts: isRecord(run.reviewArtifacts) ? run.reviewArtifacts : {},
    snapshot: normalizeRunSnapshot(snapshot, index, linkedCaptureMap, {
      aliases,
      allowLinkedCaptureFallback
    })
  }
}

function normalizeReviewBundleBody(
  bodyLike,
  { aliases = false, allowLinkedCaptureFallback = false } = {}
) {
  const body = requireRecord(bodyLike, 'review bundle')
  const bundleType = readNonEmptyString(
    firstValue(body, aliases ? ['type', 'bundleType'] : ['type'])
  )

  if (bundleType !== 'localization-review-bundle') {
    throw new Error('review bundle type must be localization-review-bundle')
  }

  const linkedCapturesSource =
    firstArray(
      body,
      aliases ? ['linkedCaptures', 'captures', 'captureShelf', 'linkedBundles'] : ['linkedCaptures']
    ) || []
  const linkedCaptures = linkedCapturesSource.map((entry, index) =>
    normalizeLinkedCaptureEntry(entry, index, { aliases })
  )
  const linkedCaptureMap = new Map(linkedCaptures.map((entry) => [entry.sourceId, entry]))
  const runsSource =
    firstArray(body, aliases ? ['runs', 'portableRuns', 'runSnapshots', 'snapshots'] : ['runs']) || []

  if (runsSource.length === 0) {
    throw new Error('review bundle must include at least one run')
  }

  const runs = runsSource.map((entry, index) =>
    normalizeRunEntry(entry, index, linkedCaptureMap, {
      aliases,
      allowLinkedCaptureFallback
    })
  )
  const compareReport = normalizeCompareReport(
    firstValue(body, aliases ? ['compareReport', 'compare', 'comparison'] : ['compareReport'])
  )
  const selection =
    firstRecord(body, aliases ? ['selection', 'compareSelection'] : ['selection']) || {}
  const createdAt = readNonEmptyString(body.createdAt) || new Date().toISOString()

  return {
    protocol: readNonEmptyString(body.protocol) || 'dreamwalker-localization-review-bundle/v1',
    type: 'localization-review-bundle',
    createdAt,
    fragmentId: readNonEmptyString(body.fragmentId),
    fragmentLabel: readNonEmptyString(body.fragmentLabel),
    selection: {
      runIds: Array.isArray(selection.runIds)
        ? selection.runIds.map((item) => readNonEmptyString(item)).filter(Boolean)
        : runs.map((run) => run.id),
      baselineRunId:
        readNonEmptyString(selection.baselineRunId) ||
        readNonEmptyString(selection.baselineId) ||
        readNonEmptyString(compareReport?.baselineRunId),
      baselineLabel:
        readNonEmptyString(selection.baselineLabel) ||
        readNonEmptyString(compareReport?.baselineLabel)
    },
    compareReport,
    artifacts: isRecord(body.artifacts) ? body.artifacts : {},
    linkedCaptures,
    runs,
    captureShelfEntries: linkedCaptures.map((entry, index) => ({
      id: deriveCaptureShelfEntryId(entry.sourceId, createdAt, index),
      label: entry.label,
      savedAt: createdAt,
      bundle: entry.bundle
    })),
    runShelfEntries: runs.map((entry) => ({
      ...entry.snapshot,
      id: entry.snapshot.id || entry.id,
      label: entry.snapshot.label || entry.label
    })),
    baselineRunId:
      readNonEmptyString(selection.baselineRunId) ||
      readNonEmptyString(selection.baselineId) ||
      readNonEmptyString(compareReport?.baselineRunId)
  }
}

class StrictCanonicalLocalizationReviewBundleImportPolicy {
  name = 'strict_canonical'
  label = 'Strict Canonical'
  style = 'exact-contract'
  tier = 'experiment'
  capabilities = {
    supportsCanonicalEnvelope: true,
    supportsLinkedCaptureFallback: false,
    supportsWrapperAliases: false,
    supportsSnapshotAliases: false
  }

  importDocument(rawDocument) {
    const parsed = requireRecord(parseRawDocument(rawDocument), 'review bundle document')
    return normalizeReviewBundleBody(parsed, {
      aliases: false,
      allowLinkedCaptureFallback: false
    })
  }
}

class LinkedCaptureFallbackLocalizationReviewBundleImportPolicy {
  name = 'linked_capture_fallback'
  label = 'Linked Capture Fallback'
  style = 'capture-aware'
  tier = 'experiment'
  capabilities = {
    supportsCanonicalEnvelope: true,
    supportsLinkedCaptureFallback: true,
    supportsWrapperAliases: false,
    supportsSnapshotAliases: false
  }

  importDocument(rawDocument) {
    const parsed = requireRecord(parseRawDocument(rawDocument), 'review bundle document')
    return normalizeReviewBundleBody(parsed, {
      aliases: false,
      allowLinkedCaptureFallback: true
    })
  }
}

class AliasFriendlyLocalizationReviewBundleImportPolicy {
  name = 'alias_friendly'
  label = 'Alias Friendly'
  style = 'compatibility-first'
  tier = 'core'
  capabilities = {
    supportsCanonicalEnvelope: true,
    supportsLinkedCaptureFallback: true,
    supportsWrapperAliases: true,
    supportsSnapshotAliases: true
  }

  importDocument(rawDocument) {
    const parsed = requireRecord(parseRawDocument(rawDocument), 'review bundle document')
    const body =
      firstRecord(parsed, ['reviewBundle', 'review', 'bundle', 'payload']) || parsed

    return normalizeReviewBundleBody(body, {
      aliases: true,
      allowLinkedCaptureFallback: true
    })
  }
}

export const EXPERIMENT_LOCALIZATION_REVIEW_BUNDLE_IMPORT_POLICIES = [
  new StrictCanonicalLocalizationReviewBundleImportPolicy(),
  new LinkedCaptureFallbackLocalizationReviewBundleImportPolicy(),
  new AliasFriendlyLocalizationReviewBundleImportPolicy()
]

export function importLocalizationReviewBundleDocument(rawDocument, policy = 'alias_friendly') {
  const policies = {
    strict_canonical: EXPERIMENT_LOCALIZATION_REVIEW_BUNDLE_IMPORT_POLICIES[0],
    linked_capture_fallback: EXPERIMENT_LOCALIZATION_REVIEW_BUNDLE_IMPORT_POLICIES[1],
    alias_friendly: EXPERIMENT_LOCALIZATION_REVIEW_BUNDLE_IMPORT_POLICIES[2]
  }
  const selectedPolicy = policies[policy]

  if (!selectedPolicy) {
    throw new Error(
      `Unsupported localization review bundle import policy: ${policy}. Expected one of ${Object.keys(
        policies
      ).join(', ')}`
    )
  }

  return selectedPolicy.importDocument(rawDocument)
}
