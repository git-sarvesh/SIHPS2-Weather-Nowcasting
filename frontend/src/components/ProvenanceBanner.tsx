/**
 * Global provenance banner.
 *
 * Pinned above every page so the experimental-data caveat is never more than one
 * glance away, regardless of which view the operator is on.
 */

import type { Provenance } from '../api/types'
import { ProvenanceBanner } from './Provenance'

export type { BannerTone } from './Provenance'
export { bannerTone, ProvenanceBanner, ProvenanceFooter, SyntheticFlag } from './Provenance'

/** Banner driven by whatever payload is currently loaded, if any. */
export function GlobalProvenanceBanner({ provenance }: { provenance?: Provenance | null }) {
  if (!provenance) return null
  return (
    <div className="px-4 pt-4 lg:px-6">
      <ProvenanceBanner provenance={provenance} />
    </div>
  )
}
