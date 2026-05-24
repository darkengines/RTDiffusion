import { map } from 'nanostores'
import type { SourceImage } from '../types'

export interface SourceState {
  root: string
  images: SourceImage[]
  selectedRel: string
  isLoading: boolean
}

export const $source = map<SourceState>({
  root: '',
  images: [],
  selectedRel: '',
  isLoading: false,
})
